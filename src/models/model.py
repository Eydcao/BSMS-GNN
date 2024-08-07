import torch
from ops import MLP, BSGMP
from utils import Normalizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BSMS_Simulator(torch.nn.Module):
    def __init__(self, cfg):
        """
        Initialize the BSMS_Simulator object.

        Parameters
        ----------
        cfg : OmegaConf
            Configuration object containing all necessary parameters.
        """
        super(BSMS_Simulator, self).__init__()
        self.cfg = cfg
        self.encode = MLP(cfg.out_dim + 1, cfg.latent_dim, cfg.latent_dim, cfg.hidden_layer, True)
        self.process = BSGMP(cfg.unet_depth, cfg.latent_dim, cfg.hidden_layer, cfg.pos_dim)
        self.decode = MLP(cfg.latent_dim, cfg.latent_dim, cfg.out_dim, cfg.hidden_layer, False)
        self.pos_dim = cfg.pos_dim
        self.device = device
        maxacum = 5e5
        self._inputNormalizer = Normalizer(cfg.out_dim + 1, max_accumulations=maxacum, device=device, name="in_norm")
        self._targetNormalizer = Normalizer(cfg.out_dim, max_accumulations=maxacum, device=device, name="out_norm")

    def _get_nodal_latent_input(self, node_in):
        """
        Remove the last pos_dim + 1 from the input tensor.

        Parameters
        ----------
        node_in : torch.Tensor
            Input tensor of shape (B, N, C + pos_dim + 1), where the last pos_dim + 1 is mesh_pos and node_type.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, N, C+1) with the mesh_pos removed.
        """
        x = node_in[..., : -1 - self.pos_dim].clone()
        node_type = node_in[..., -1:].clone()
        # cat
        return torch.cat([x, node_type], dim=-1)

    def _get_pos_type(self, node_in):
        """
        Return mesh_pos and node_type.

        Parameters
        ----------
        node_in : torch.Tensor
            Input tensor of shape (B, N, C + pos_dim + 1).

        Returns
        -------
        tuple
            mesh_pos (B, N, pos_dim), node_type (B, N)
        """
        return node_in[..., -(1 + self.pos_dim) : -1].clone(), node_in[..., -1].clone()

    def _deltas(self, node_in, node_tar):
        """
        Compute the difference between target and input nodes.

        Parameters
        ----------
        node_in : torch.Tensor
            Input tensor of shape (B, N, C+1).
            Where the last dimension is the node type, no need to calculate delta
        node_tar : torch.Tensor
            Target tensor of shape (B, N, C).

        Returns
        -------
        torch.Tensor
            Delta tensor of shape (B, N, C)
        """
        return node_tar - node_in[..., : node_tar.shape[-1]]

    def _encode_process_decode(self, node_feature, m_ids, multi_gs, pos):
        """
        Encode, process, and decode node features.

        Parameters
        ----------
        node_feature : torch.Tensor
            Node feature tensor of shape (B, N, C).
        m_ids : torch.Tensor
            Multi-layer IDs.
        multi_gs : list
            Multi-layer graphs.
        pos : torch.Tensor
            Node positions of shape (B, N, pos_dim).

        Returns
        -------
        torch.Tensor
            Processed node features of shape (B, N, C)
        """
        x = self.encode(node_feature)
        x = self.process(x, m_ids, multi_gs, pos)
        x = self.decode(x)
        return x

    def _warmup(self, node_in, node_tar):
        """
        Warmup phase to accumulate normalizer statistics.

        Parameters
        ----------
        node_in : torch.Tensor
            Input tensor of shape (B, N, C + pos_dim + 1).
        node_tar : torch.Tensor
            Target tensor of shape (B, N, C).
        """
        node_in = self._get_nodal_latent_input(node_in)
        target_delta = self._deltas(node_in, node_tar)
        # accumulate normalizer
        self._inputNormalizer(node_in, accumulate=True)
        self._targetNormalizer(target_delta, accumulate=True)

        return node_tar.new_zeros(node_tar.shape)

    def _forward(self, m_ids, m_gs, node_in, node_mask):
        """
        Forward pass for the model.

        Parameters
        ----------
        m_ids : torch.Tensor
            Multi-layer IDs.
        m_gs : list
            Multi-layer graphs.
        node_in : torch.Tensor
            Input tensor of shape (B, N, C + pos_dim + 1).
        node_tar : torch.Tensor
            Target tensor of shape (B, N, C).

        Returns
        -------
        tuple
            Average MSE, predicted target, and number of non-zero elements.
        """
        node_pos, node_type = self._get_pos_type(node_in)

        node_in = self._get_nodal_latent_input(node_in)
        # target_delta = self._deltas(node_in, node_tar)

        # normalize input and target_delta
        norm_node_in = self._inputNormalizer(node_in, accumulate=False)
        # norm_target_delta = self._targetNormalizer(target_delta, accumulate=False)

        # infer: encode->MP->decode->time integrate to update states
        norm_pred_delta = self._encode_process_decode(norm_node_in, m_ids, m_gs, node_pos)

        # get original unit pred
        pred_delta = self._targetNormalizer.inverse(norm_pred_delta)
        # set none mask area to 0
        pred_delta = pred_delta * node_mask
        pred_target = node_in[..., : pred_delta.shape[-1]] + pred_delta
        return pred_target

    def forward(self, data, consistent_mesh, warmup):
        """
        Forward pass interface.

        Parameters
        ----------
        m_ids : torch.Tensor
            Multi-layer IDs.
        m_gs : list
            Multi-layer graphs.
        node_in : torch.Tensor
            Input tensor of shape (B, N, C + pos_dim + 1).
        node_tar : torch.Tensor
            Target tensor of shape (B, N, C).
        warmup : bool
            Flag for warmup phase.

        Returns
        -------
        tuple or None
            If warmup, returns (None, None, None). Otherwise, returns the result of _forward.
        """
        # unpack the data
        if consistent_mesh:
            node_in, node_tar, node_mask, m_gs, m_ids = data
            m_gs = [g[0] for g in m_gs]
            m_ids = [i[0] for i in m_ids]
        else:
            node_in, node_tar, node_mask = data[0].x, data[0].y, data[0].mask
            # make len(shape)==3, so later code is consistent
            node_in = node_in.unsqueeze(0)
            node_tar = node_tar.unsqueeze(0)
            node_mask = node_mask.unsqueeze(0)
            m_gs = [d.edge_index for d in data]
            m_ids = [data[i].face for i in range(len(m_gs) - 1)]
        if warmup:
            dummy_zero = self._warmup(node_in, node_tar)
            # # report the updated mean and std of normlizers
            # self._inputNormalizer.report()
            # self._targetNormalizer.report()
            return dummy_zero
        else:
            return self._forward(m_ids, m_gs, node_in, node_mask)
