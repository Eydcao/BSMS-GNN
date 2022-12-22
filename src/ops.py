import torch.nn as nn
import torch
from torch_geometric.utils import degree
from torch.nn import Sequential as Seq, Linear, ReLU, LayerNorm
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


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


class GMP(MessagePassing):
    def __init__(self, latent_dim, hidden_layer, pos_dim, lagrangian):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_node_delta = MLP(2 * latent_dim, latent_dim, latent_dim, hidden_layer, True)
        edge_info_in_len = 2 * latent_dim + 2 * pos_dim + 2 if lagrangian else 2 * latent_dim + pos_dim + 1
        self.mlp_edge_info = MLP(edge_info_in_len, latent_dim, latent_dim, hidden_layer, True)
        self.lagrangian = lagrangian
        self.pos_dim = pos_dim

    def forward(self, x, g, pos):
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            T, _, _ = x.shape
            x_i = x[:, i]
            x_j = x[:, j]
        elif len(x.shape) == 2:
            x_i = x[i]
            x_j = x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        if len(pos.shape) == 3:
            pi = pos[:, i]
            pj = pos[:, j]
        elif len(pos.shape) == 2:
            pi = pos[i]
            pj = pos[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        dir = pi - pj  # in shape (T),N,dim
        if self.lagrangian:
            norm_w = torch.norm(dir[..., :self.pos_dim], dim=-1, keepdim=True)  # in shape (T),N,1
            norm_m = torch.norm(dir[..., self.pos_dim:], dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, norm_w, norm_m], dim=-1)
        else:
            norm = torch.norm(dir, dim=-1, keepdim=True)  # in shape (T),N,1
            fiber = torch.cat([dir, norm], dim=-1)

        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
        edge_embedding = self.mlp_edge_info(tmp)

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")

        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_node_delta(tmp) + x


class WeightedEdgeConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add', flow='target_to_source')

    def forward(self, x, g, ew, aggragating=True):
        # aggregating: False means returning
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            weighted_info = x[:, i] if aggragating else x[:, j]
        elif len(x.shape) == 2:
            weighted_info = x[i] if aggragating else x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        weighted_info *= ew.unsqueeze(-1)
        target_index = j if aggragating else i
        aggr_out = scatter(weighted_info, target_index, dim=-2, dim_size=x.shape[-2], reduce="sum")
        return aggr_out

    @torch.no_grad()
    def cal_ew(self, w, g):
        deg = degree(g[0], dtype=torch.float, num_nodes=w.shape[0])
        normed_w = w.squeeze(-1) / deg
        i = g[0]
        j = g[1]
        w_to_send = normed_w[i]
        eps = 1e-12
        aggr_w = scatter(w_to_send, j, dim=-1, dim_size=normed_w.size(0), reduce="sum") + eps
        ec = w_to_send / aggr_w[j]
        return ec, aggr_w


class GMPEdgeAggregatedRes(MessagePassing):
    def __init__(self, in_dim, latent_dim, hidden_layer):
        super().__init__(aggr='add', flow='target_to_source')
        self.mlp_edge_info = MLP(in_dim, latent_dim, latent_dim, hidden_layer, True)

    def forward(self, x, g, pos, pos_w, use_mat=True, use_world=True):
        i = g[0]
        j = g[1]
        if len(x.shape) == 3:
            T, _, _ = x.shape
            x_i = x[:, i]
            x_j = x[:, j]
        elif len(x.shape) == 2:
            x_i = x[i]
            x_j = x[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        if len(pos.shape) == 3:
            if use_mat:
                pi = pos[:, i]
                pj = pos[:, j]
            if use_world:
                pwi = pos_w[:, i]
                pwj = pos_w[:, j]
        elif len(pos.shape) == 2:
            if use_mat:
                pi = pos[i]
                pj = pos[j]
            if use_world:
                pwi = pos_w[i]
                pwj = pos_w[j]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")
        if use_mat:
            dir = pi - pj  # in shape (T),N,dim
            norm = torch.norm(dir, dim=-1, keepdim=True)  # in shape (T),N,1
        if use_world:
            dir_w = pwi - pwj  # in shape (T),N,dim
            norm_w = torch.norm(dir_w, dim=-1, keepdim=True)  # in shape (T),N,1

        if use_mat and use_world:
            fiber = torch.cat([dir, norm, dir_w, norm_w], dim=-1)
        elif not use_mat and use_world:
            fiber = torch.cat([dir_w, norm_w], dim=-1)
        elif use_mat and not use_world:
            fiber = torch.cat([dir, norm], dim=-1)
        else:
            raise NotImplementedError("at least one pos needs to cal fiber info")

        if len(x.shape) == 3 and len(pos.shape) == 2:
            tmp = torch.cat([fiber.unsqueeze(0).repeat(T, 1, 1), x_i, x_j], dim=-1)
        else:
            tmp = torch.cat([fiber, x_i, x_j], dim=-1)
        edge_embedding = self.mlp_edge_info(tmp)

        aggr_out = scatter(edge_embedding, j, dim=-2, dim_size=x.shape[-2], reduce="sum")

        return aggr_out


class ContactGMP(nn.Module):
    def __init__(self, latent_dim, hidden_layer, pos_dim, lagrangian, MP_model=GMPEdgeAggregatedRes):
        super(ContactGMP, self).__init__()
        in_dim_main = 2 * latent_dim + 2 * (pos_dim + 1)
        in_dim_cont = 2 * latent_dim + pos_dim + 1
        in_dim_node = 3 * latent_dim
        self.mp_main = MP_model(in_dim_main, latent_dim, hidden_layer)
        self.mp_cont = MP_model(in_dim_cont, latent_dim, hidden_layer)
        self.mlp_node_delta = MLP(in_dim_node, latent_dim, latent_dim, hidden_layer, True)

    def forward(self, x, g_cg, pos_posw):
        g = g_cg[0]
        cg = g_cg[1]
        D = int(pos_posw.shape[-1] // 2)
        if len(pos_posw.shape) == 2:
            pos = pos_posw[:, :D]
            pos_w = pos_posw[:, D:]
        elif len(pos_posw.shape) == 3:
            pos = pos_posw[..., :D]
            pos_w = pos_posw[..., D:]
        else:
            raise NotImplementedError("Only implemented for dim 2 and 3")

        agg_main = self.mp_main(x, g, pos, pos_w, use_mat=True, use_world=True)
        agg_cont = self.mp_cont(x, cg, pos, pos_w, use_mat=False, use_world=True)
        tmp = torch.cat([x, agg_main, agg_cont], dim=-1)

        return self.mlp_node_delta(tmp) + x


class Unpool(nn.Module):
    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, h, pre_node_num, idx):
        if len(h.shape) == 2:
            new_h = h.new_zeros([pre_node_num, h.shape[-1]])
            new_h[idx] = h
        elif len(h.shape) == 3:
            new_h = h.new_zeros([h.shape[0], pre_node_num, h.shape[-1]])
            new_h[:, idx] = h
        return new_h
