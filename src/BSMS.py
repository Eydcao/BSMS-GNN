import torch.nn as nn
import torch
from ops import GMP, Unpool, WeightedEdgeConv


class BSGMP(nn.Module):

    def __init__(self, l_n, ld, hidden_layer, pos_dim, lagrangian, MP_model=GMP, edge_set_num=1):
        super(BSGMP, self).__init__()
        self.bottom_gmp = MP_model(ld, hidden_layer, pos_dim, lagrangian)
        self.down_gmps = nn.ModuleList()
        self.up_gmps = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = l_n
        self.edge_conv = WeightedEdgeConv()
        for _ in range(self.l_n):
            self.down_gmps.append(MP_model(ld, hidden_layer, pos_dim, lagrangian))
            self.up_gmps.append(MP_model(ld, hidden_layer, pos_dim, lagrangian))
            self.unpools.append(Unpool())
        self.esn = edge_set_num
        self.lagrangian = lagrangian

    def forward(self, h, m_ids, m_gs, pos, weights=None):
        # h is in shape of (T), N, F
        # if edge_set_num>1, then m_g is in shape: Level,(Set),2,Edges, the 0th Set is main/material graph
        # pos is in (T),N,D
        down_outs = []
        down_ps = []
        cts = []
        hs = []
        w = pos.new_ones((pos.shape[-2], 1)) if weights is None else weights
        # down pass
        for i in range(self.l_n):
            h = self.down_gmps[i](h, m_gs[i], pos)
            if i == 0 and self.lagrangian:
                h = self.down_gmps[i](h, m_gs[i], pos)
            # record the infor before aggregation
            down_outs.append(h)
            down_ps.append(pos)
            # aggregate then pooling
            # cal edge_weights
            tmp_g = m_gs[i][0] if self.esn > 1 else m_gs[i]
            ew, w = self.edge_conv.cal_ew(w, tmp_g)
            h = self.edge_conv(h, tmp_g, ew)
            pos = self.edge_conv(pos, tmp_g, ew)
            cts.append(ew)
            # pooling
            if len(h.shape) == 3:
                h = h[:, m_ids[i]]
            elif len(h.shape) == 2:
                h = h[m_ids[i]]
            if len(pos.shape) == 3:
                pos = pos[:, m_ids[i]]
            elif len(pos.shape) == 2:
                pos = pos[m_ids[i]]
            w = w[m_ids[i]]
        # bottom pass
        h = self.bottom_gmp(h, m_gs[self.l_n], pos)
        if self.lagrangian:
            h = self.bottom_gmp(h, m_gs[self.l_n], pos)
        # up pass
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = m_gs[up_idx], m_ids[up_idx]
            h = self.unpools[i](h, down_outs[up_idx].shape[-2], idx)
            tmp_g = g[0] if self.esn > 1 else g
            h = self.edge_conv(h, tmp_g, cts[up_idx], aggragating=False)
            h = self.up_gmps[i](h, g, down_ps[up_idx])
            if up_idx == 0 and self.lagrangian:
                h = self.up_gmps[i](h, g, down_ps[up_idx])
            h = h.add(down_outs[up_idx])

        return h
