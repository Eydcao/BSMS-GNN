import torch
from ops import GMP, MLP, ContactGMP
from BSMS import BSGMP


class ModelGeneral(torch.nn.Module):
    def __init__(self, pos_dim, in_dim, out_dim, ld, layer_num, mlp_hidden_layer, MP_times, lagrangian, MP_model, edge_set_num, has_contact):
        super(ModelGeneral, self).__init__()
        self.encode = MLP(in_dim, ld, ld, mlp_hidden_layer, True)
        self.process = BSGMP(layer_num, ld, mlp_hidden_layer, pos_dim, lagrangian, MP_model, edge_set_num)
        self.decode = MLP(ld, ld, out_dim, mlp_hidden_layer, False)
        self.MP_times = MP_times
        self.pos_dim = pos_dim
        self.mse = torch.nn.MSELoss(reduction='none')

    def _get_nodal_latent_input(self, node_in):
        # NOTE implement in childs
        # NOTE we want to remove absolute position from input
        return node_in

    def _get_pos_type(self, node_in):
        # NOTE by defualt, we agree in feature ends with X,type
        return node_in[..., -(1 + self.pos_dim):-1].clone(), node_in[..., -1].clone()

    def _penalize(self, loss, pen_coeff):
        # NOTE implement in childs, pen_coeff shape should be [B(or 1),F] or [F]
        # loss in [B,N,F]
        if len(pen_coeff.shape) == 2:
            pen_coeff = pen_coeff.unsqueeze(1)
        return loss * pen_coeff

    def _update_states(self, node_in, node_tar, node_type, out):
        # NOTE implement in childs
        return out

    def _pre(self, node_in, node_tar, node_type):
        # NOTE implement in childs
        return node_in

    def _mask(self, node_in, node_tar, node_type, out):
        # NOTE implement in childs
        mask = torch.ones_like(node_tar)
        return out, mask

    def _EMD(self, node_feature, m_ids, multi_gs, pos):
        node_feature = self._get_nodal_latent_input(node_feature)
        x = self.encode(node_feature)
        for _ in range(self.MP_times):
            x = self.process(x, m_ids, multi_gs, pos)
        x = self.decode(x)
        return x

    def forward(self, m_idx, m_gs, node_in, node_tar, pen_coeff=None):
        # get mat pos and type
        node_pos, node_type = self._get_pos_type(node_in)
        # preprocess: set scripted bcs
        node_in = self._pre(node_in, node_tar, node_type)
        # infer: encode->MP->decode->time integrate to update states
        out = self._EMD(node_in, m_idx, m_gs, node_pos)
        out = self._update_states(node_in, node_tar, node_type, out)
        # masking: e.g. 1st kind bc, scripted bc
        out, mask = self._mask(node_in, node_tar, node_type, out)
        # error cal
        loss = self.mse(out, node_tar)
        if pen_coeff != None:
            loss = self._penalize(loss, pen_coeff)
        loss = (loss * mask).sum()
        non_zero_elements = mask.sum()
        mse_loss_val = loss / non_zero_elements

        return mse_loss_val, out, non_zero_elements


class Cylinder(ModelGeneral):
    def __init__(self, pos_dim, ld, layer_num, mlp_hidden_layer, MP_times):
        in_dim = 1 + pos_dim  #vel,type
        out_dim = pos_dim  #vel
        MP_model = GMP
        super(Cylinder, self).__init__(pos_dim, in_dim, out_dim, ld, layer_num, mlp_hidden_layer, MP_times, lagrangian=False, MP_model=MP_model, edge_set_num=1, has_contact=False)

    def _update_states(self, node_in, node_tar, node_type, out):
        out += node_in[..., :self.pos_dim]
        return out

    def _pre(self, node_in, node_tar, node_type):
        # (0 int nodes; 5 outlet nodes)
        preset_node = ((node_type != 0) * (node_type != 5)).bool().unsqueeze(-1)
        node_in[..., :self.pos_dim] = torch.where(preset_node, node_tar[..., :self.pos_dim], node_in[..., :self.pos_dim])
        return node_in

    def _mask(self, node_in, node_tar, node_type, out):
        # (0 int nodes; 5 outlet nodes)
        measure_nodes = ((node_type == 0) + (node_type == 5)).bool().unsqueeze(-1)
        mask = torch.where(measure_nodes, torch.ones_like(node_tar), torch.zeros_like(node_tar))
        out = torch.where(measure_nodes, out, node_tar)
        return out, mask

    def _get_nodal_latent_input(self, node_in):
        # in_dim for nodal encoding: [vel,type] out of [vel,X,type]
        return torch.cat((node_in[..., :self.pos_dim], node_in[..., -1:]), dim=-1)


class Airfoil(ModelGeneral):
    def __init__(self, pos_dim, ld, layer_num, mlp_hidden_layer, MP_times):
        in_dim = 2 + pos_dim  # vel,density,type
        out_dim = pos_dim + 2  # vel,density,pressure
        MP_model = GMP
        super(Airfoil, self).__init__(pos_dim, in_dim, out_dim, ld, layer_num, mlp_hidden_layer, MP_times, lagrangian=False, MP_model=MP_model, edge_set_num=1, has_contact=False)

    def _update_states(self, node_in, node_tar, node_type, out):
        # donot time integrate pressure
        out[..., :-1] = out[..., :-1] + node_in[..., :-(1 + self.pos_dim)]
        return out

    def _pre(self, node_in, node_tar, node_type):
        # 0 int node
        preset_node = (node_type != 0).bool().unsqueeze(-1)
        node_in[..., :self.pos_dim + 1] = torch.where(preset_node, node_tar[..., :self.pos_dim + 1], node_in[..., :self.pos_dim + 1])
        return node_in

    def _mask(self, node_in, node_tar, node_type, out):
        # 0 int node
        int_node = (node_type == 0).bool().unsqueeze(-1)
        mask = torch.where(int_node, torch.ones_like(node_tar), torch.zeros_like(node_tar))
        out = torch.where(int_node, out, node_tar)
        return out, mask

    def _get_nodal_latent_input(self, node_in):
        # in_dim for nodal encoding: [vel,density,type] out of [vel,density,pos,type]
        return torch.cat((node_in[..., :self.pos_dim + 1], node_in[..., -1:]), dim=-1)


class Plate(ModelGeneral):
    def __init__(self, pos_dim, ld, layer_num, mlp_hidden_layer, MP_times):
        # in: d_x(used for driven nodes only),type
        # out: d_x
        in_dim = 1 + pos_dim
        out_dim = pos_dim
        MP_model = ContactGMP
        super(Plate, self).__init__(pos_dim, in_dim, out_dim, ld, layer_num, mlp_hidden_layer, MP_times, lagrangian=True, MP_model=MP_model, edge_set_num=2, has_contact=True)

    def _get_pos_type(self, node_in):
        # in: x,X,d_x(used for driven nodes only),type
        pos_mat_world = torch.cat((node_in[..., self.pos_dim:2 * self.pos_dim], node_in[..., :self.pos_dim]), dim=-1)
        node_type = node_in[..., -1].clone()
        return pos_mat_world, node_type

    def _update_states(self, node_in, node_tar, node_type, out):
        out = out + node_in[..., :self.pos_dim]  # out: no stress; in: no mat pos and type
        return out

    def _pre(self, node_in, node_tar, node_type):
        # NOTE 1. only change pos for fixed nodes (type 3)
        #      2. for driven nodes (type 1), record the delta pos info and prepend before type
        # NOTE: why include vel,
        #       because if we move the driven node directly, the fiber direction is wrong;
        #       else if we dont move them, how the solver know its moving?
        fix_node = (node_type == 3).bool().unsqueeze(-1)
        drive_node = (node_type == 1).bool().unsqueeze(-1)
        node_in[..., :self.pos_dim] = torch.where(fix_node, node_tar[..., :self.pos_dim], node_in[..., :self.pos_dim])
        delta_x_driven = node_tar[..., :self.pos_dim] - node_in[..., :self.pos_dim]
        delta_x_driven = torch.where(drive_node, delta_x_driven, torch.zeros_like(delta_x_driven))
        # resemble to x,X,delta_x,type
        node_in = torch.cat((node_in[..., :2 * self.pos_dim], delta_x_driven, node_in[..., -1].unsqueeze(-1)), dim=-1)
        return node_in

    def _mask(self, node_in, node_tar, node_type, out):
        # only measure int nodes(0)
        int_node = (node_type == 0).bool().unsqueeze(-1)
        mask = torch.where(int_node, torch.ones_like(node_tar), torch.zeros_like(node_tar))
        out = torch.where(int_node, out, node_tar)
        return out, mask

    def _get_nodal_latent_input(self, node_in):
        # in_dim for nodal encoding: [vel(used for driven nodes only),type] out of [x,X,vel(used for driven nodes only),type]
        return node_in[..., -(1 + self.pos_dim):].clone()


class Font(ModelGeneral):
    def __init__(self, pos_dim, ld, layer_num, mlp_hidden_layer, MP_times):
        # in: type
        # out: d_x
        in_dim = 1
        out_dim = pos_dim
        MP_model = ContactGMP
        super(Font, self).__init__(pos_dim, in_dim, out_dim, ld, layer_num, mlp_hidden_layer, MP_times, lagrangian=True, MP_model=MP_model, edge_set_num=2, has_contact=True)

    def _get_pos_type(self, node_in):
        pos_mat_world = torch.cat((node_in[..., self.pos_dim:2 * self.pos_dim], node_in[..., :self.pos_dim]), dim=-1)
        node_type = node_in[..., -1].clone()
        return pos_mat_world, node_type

    def _update_states(self, node_in, node_tar, node_type, out):
        out = out + node_in[..., :self.pos_dim]
        return out

    def _get_nodal_latent_input(self, node_in):
        # in_dim for nodal encoding: [type] out of [x,X,type]
        return node_in[..., -1:].clone()
