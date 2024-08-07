import torch
from utils import degree, scatter_sum
from torch.nn import Sequential as Seq, Linear, ReLU, LayerNorm


class MLP(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, hidden_layers, layer_normalized=True):
        super(MLP, self).__init__()
        modules = []
        for l in range(hidden_layers):
            if l == 0:
                modules.append(Linear(input_dim, latent_dim))
            else:
                modules.append(Linear(latent_dim, latent_dim))
            modules.append(ReLU())
        modules.append(Linear(latent_dim, output_dim))
        if layer_normalized:
            modules.append(LayerNorm(output_dim, elementwise_affine=False))

        self.seq = Seq(*modules)

    def forward(self, x):
        return self.seq(x)


class GMP(torch.nn.Module):
    """The copied Graph Message Passing (GMP) block from BSMS GNN."""

    def __init__(self, latent_dim, hidden_layer, pos_dim):
        """
        Initialize the GMP block.

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space.
        hidden_layer : int
            Number of hidden layers.
        pos_dim : int
            Dimension of the positional encoding.
        """
        super(GMP, self).__init__()
        self.mlp_node = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer)
        edge_info_in_len = 2 * latent_dim + pos_dim + 1
        self.mlp_edge = MLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer)
        self.pos_dim = pos_dim

    def forward(self, x, g, pos):
        """
        Forward pass for GMP block.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape [B, N, C] or [N, C].
        g : torch.Tensor
            Graph connectivity (edges) of shape [2, E].
        pos : torch.Tensor
            Node positional information of shape [B, N, pos_dim] or [N, pos_dim].

        Returns
        -------
        torch.Tensor
            Updated node features.
        """
        i, j = g[0], g[1]

        if len(x.shape) == 3:
            B, _, _ = x.shape
            x_i, x_j = x[:, i], x[:, j]
        elif len(x.shape) == 2:
            x_i, x_j = x[i], x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")

        if len(pos.shape) == 3:
            pi, pj = pos[:, i], pos[:, j]
        elif len(pos.shape) == 2:
            pi, pj = pos[i], pos[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")

        dir = pi - pj  # (B, N, pos_dim) or (N, pos_dim)
        norm = torch.norm(dir, dim=-1, keepdim=True)  # (B, N, 1) or (N, 1)
        fiber = torch.cat([dir, norm], dim=-1)  # (B, N, pos_dim+1) or (N, pos_dim+1)
        # below is the cat between fiber and node latent features
        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(B, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
        # get the information flow on the edge
        edge_embedding = self.mlp_edge(tmp)
        # sum the edge information to the in node
        aggr_out = scatter_sum(edge_embedding, j, dim=-2, dim_size=x.shape[-2])

        # MLP take input as the cat between x and the aggregated edge information flow
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_node(tmp) + x


class WeightedEdgeConv(torch.nn.Module):
    """Weighted Edge Convolution layer for transition between layers."""

    def __init__(self, *args):
        super(WeightedEdgeConv, self).__init__()

    def forward(self, x, g, ew, aggragating=True):
        """
        Forward pass for WeightedEdgeConv layer.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape [B, N, C] or [N, C].
        g : torch.Tensor
            Graph connectivity (edges) of shape [2, E].
        ew : torch.Tensor
            Edge weights for convolution of shape [E].
        aggragating : bool, optional
            If True, aggregate messages (used in down pass); if False, return messages (used in up pass).

        Returns
        -------
        torch.Tensor
            Aggregated or scattered node features.
        """
        i, j = g[0], g[1]

        if len(x.shape) == 3:
            weighted_info = x[:, i] if aggragating else x[:, j]
        elif len(x.shape) == 2:
            weighted_info = x[i] if aggragating else x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")

        weighted_info *= ew.unsqueeze(-1)
        target_index = j if aggragating else i
        aggr_out = scatter_sum(weighted_info, target_index, dim=-2, dim_size=x.shape[-2])

        return aggr_out

    @torch.no_grad()
    def cal_ew(self, w, g):
        """
        Calculate the edge weights for later use in forward.

        Parameters
        ----------
        w : torch.Tensor
            Node weights of shape [N, 1].
        g : torch.Tensor
            Graph connectivity (edges) of shape [2, E].

        Returns
        -------
        tuple
            Edge weights for convolution and aggregated node weights (used for iteratively calculating this in the next layer).
        """
        deg = degree(g[0], dtype=torch.float, num_nodes=w.shape[0])
        normed_w = w.squeeze(-1) / deg
        i, j = g[0], g[1]
        w_to_send = normed_w[i]
        eps = 1e-12
        aggr_w = scatter_sum(w_to_send, j, dim=-1, dim_size=normed_w.size(0)) + eps
        ec = w_to_send / aggr_w[j]

        return ec, aggr_w


class Unpool(torch.nn.Module):
    """Unpooling layer for graph neural networks."""

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, h, pre_node_num, idx):
        """
        Forward pass for the unpooling layer.

        Parameters
        ----------
        h : torch.Tensor
            Node features of shape [N, C] or [B, N, C].
        pre_node_num : int
            Number of nodes in the previous upper layer.
        idx : torch.Tensor
            Relative indices (in the previous upper layer) for unpooling of shape [N] or [B, N].

        Returns
        -------
        torch.Tensor
            Unpooled node features of shape [pre_node_num, C] or [B, pre_node_num, C].
        """
        if len(h.shape) == 2:
            new_h = h.new_zeros([pre_node_num, h.shape[-1]])
            new_h[idx] = h
        elif len(h.shape) == 3:
            new_h = h.new_zeros([h.shape[0], pre_node_num, h.shape[-1]])
            new_h[:, idx] = h

        return new_h
