import os
import pickle
import numpy as np
import h5py
import torch
from torch_geometric.data import Data, Dataset
import json
from enum import Enum
from helpers_mesh import tetras_to_edges, triangles_to_edges, quads_to_edges, lines_to_edges
from helpers_convert import _flat_edge_to_adj_mat
from helpers_bistride import generate_multi_layer_stride, multi_layer_contact_edge, _remove_invalid_connection, SeedingHeuristic
from helpers_contact import contact_edge_no_self, contact_edge


class MeshType(Enum):
    Triangle = 1
    Tetrahedron = 2
    Quad = 3
    Line = 4
    Flat = 5


class MeshGeneralDataset(Dataset):
    def __init__(self,
                 root,
                 in_normal_feature_list,
                 out_normal_feature_list,
                 roll_normal_feature_list,
                 instance_id,
                 layer_num,
                 stride,
                 mode,
                 noise_shuffle,
                 noise_level,
                 noise_gamma,
                 recal_mesh,
                 consist_mesh,
                 mesh_type,
                 has_contact,
                 has_self_contact,
                 dirichelet_markers=[],
                 seed_heuristic=SeedingHeuristic.MinAve):
        # NOTE instance_id for a specific transient seq; each instance is in shape T,N,F
        self.instance_id = instance_id
        self.mode = mode
        self.data_dir = os.path.join(root, 'outputs_' + mode + '/')
        self.layer_num = layer_num
        self.recal_mesh = recal_mesh
        self.consist_mesh = consist_mesh
        self.mesh_type = mesh_type
        self.has_contact = has_contact
        self.has_self_contact = has_self_contact
        self.dirichelet_markers = dirichelet_markers
        self.seed_heuristic = seed_heuristic
        # read all features indicated in meta
        with open(os.path.join(root, 'meta.json'), 'r') as fp:
            self.meta = json.loads(fp.read())
        field_names = self.meta['field_names']
        fields = dict()
        with h5py.File(os.path.join(self.data_dir, str(instance_id) + '.h5'), 'r') as f:
            for name in field_names:
                if name == "cells":
                    fields[name] = np.array(f[name])
                    self.cells = fields[name][0]
                else:
                    fields[name] = torch.tensor(np.array(f[name]), dtype=torch.float)
        # read normalization info
        self._read_normalization_info(in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list)
        # shuffle noise and enhance data, determine do or not
        if noise_level is None or not noise_shuffle:
            self.noise_shuffle = False
            self.noise_shuffle = None
            self.noise_gamma = 1.0
        else:
            self.noise_shuffle = True
            self.noise_level = torch.tensor(noise_level, dtype=torch.float)
            self.noise_gamma = noise_gamma
        # cal len of dataset
        self.stride = stride
        self.strided_idx = list(range(0, fields["mesh_pos"].shape[0], stride))
        self.L = len(self.strided_idx) - 1
        for name in field_names:
            fields[name] = fields[name][self.strided_idx]
        # re-cal multi level mesh?
        self._cal_multi_mesh(fields)
        in_feature, tar_feature = self._preprocess(fields)
        # normalization
        self.in_feature, self.tar_feature = self._normalize(in_feature, tar_feature)
        super().__init__(root)

    def len(self):
        return self.L

    def get(self, idx):
        # idx in time seq (enhanced by noise shuffle)
        data = Data(x=self.in_feature[idx], y=self.tar_feature[idx])
        return data

    def _read_normalization_info(self, in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list):
        # collect in normalization
        for i, fea in enumerate(in_normal_feature_list):
            temp_std = torch.tensor(self.meta['normalization_info'][fea]['std'], dtype=torch.float)
            temp_mean = torch.tensor(self.meta['normalization_info'][fea]['mean'], dtype=torch.float)
            if i == 0:
                self.std_in = temp_std
                self.mean_in = temp_mean
            else:
                self.std_in = torch.cat((self.std_in, temp_std), dim=-1)
                self.mean_in = torch.cat((self.mean_in, temp_mean), dim=-1)
        # collect out normalization
        for i, fea in enumerate(out_normal_feature_list):
            temp_std = torch.tensor(self.meta['normalization_info'][fea]['std'], dtype=torch.float)
            temp_mean = torch.tensor(self.meta['normalization_info'][fea]['mean'], dtype=torch.float)
            if i == 0:
                self.std_out = temp_std
                self.mean_out = temp_mean
            else:
                self.std_out = torch.cat((self.std_out, temp_std), dim=-1)
                self.mean_out = torch.cat((self.mean_out, temp_mean), dim=-1)
        # collect roll-out normalization
        self.roll_l = 0
        for i, fea in enumerate(roll_normal_feature_list):
            temp_std = torch.tensor(self.meta['normalization_info'][fea]['std'], dtype=torch.float)
            self.roll_l += temp_std.shape[-1]
        # NOTE assume/let all leading features align with the list ordering here
        self.in_norm_l = self.std_in.shape[0]
        self.out_norm_l = self.std_out.shape[0]

    def _normalize(self, t_in, t_out):
        x_in = t_in.clone()
        x_out = t_out.clone()
        x_in[..., :self.in_norm_l] = (x_in[..., :self.in_norm_l] - self.mean_in) / self.std_in
        x_out[..., :self.out_norm_l] = (x_out[..., :self.out_norm_l] - self.mean_out) / self.std_out
        return x_in, x_out

    def _unnormalize(self, t_in, t_out):
        x_in = t_in.clone()
        x_out = t_out.clone()
        x_in[..., :self.in_norm_l] = x_in[..., :self.in_norm_l] * self.std_in + self.mean_in
        x_out[..., :self.in_norm_l] = x_out[..., :self.in_norm_l] * self.std_out + self.mean_out
        return x_in, x_out

    def _push_forward(self, out, current_stat):
        current_stat[..., :self.roll_l] = out[..., :self.roll_l]
        return current_stat

    def suggested_pen_coef(self):
        return self.std_out * self.std_out

    def _preprocess(self, fields):
        # NOTE implement in child class
        raise NotImplementedError("This needs to be implemented")

    def _cal_mesh(self, fields):
        if self.consist_mesh:
            mmfile = os.path.join(self.data_dir, 'mmesh_flat.dat')
        else:
            mmfile = os.path.join(self.data_dir, str(self.instance_id) + '_mmesh_flat.dat')
        mmexist = os.path.isfile(mmfile)
        if self.recal_mesh or not mmexist:
            m_mesh = dict()
            # normal edge
            if self.mesh_type == MeshType.Triangle:
                edge_i = triangles_to_edges(self.cells)
            if self.mesh_type == MeshType.Tetrahedron:
                edge_i = tetras_to_edges(self.cells)
            if self.mesh_type == MeshType.Quad:
                edge_i = quads_to_edges(self.cells)
            if self.mesh_type == MeshType.Line:
                edge_i = lines_to_edges(self.cells)
            if self.mesh_type == MeshType.Flat:
                edge_i = self.cells
            # contact edge
            if self.has_contact:
                self.contact_radius = self.meta['collision_radius']
                w_pos = fields['world_pos']
                if self.has_self_contact:
                    init_contact_mat = contact_edge(w_pos[:-1], edge_i, self.contact_radius)
                else:
                    init_contact_mat = contact_edge_no_self(w_pos[:-1], edge_i, self.contact_radius)
                # record
                # treat like a zero-level unet, so that the batch merge function (for mgs) works properly
                # NOTE remember to get the m_g[0] in MPs
                init_contact_mat = [[g] for g in init_contact_mat]
                m_mesh['m_cgs'] = init_contact_mat
            # remove this after the contact edge calculation, otherwise too slow
            # treat like a zero-level unet, so that the batch merge function (for mgs) works properly
            # NOTE remember to get the m_g[0] in MPs
            edge_i = [_remove_invalid_connection(edge_i, fields['node_type'][0], self.dirichelet_markers)]
            # record
            m_mesh['m_gs'] = edge_i
            # dump
            pickle.dump(m_mesh, open(mmfile, 'wb'))
        else:
            m_mesh = pickle.load(open(mmfile, 'rb'))
            edge_i = m_mesh['m_gs']
            if self.has_contact:
                init_contact_mat = m_mesh['m_cgs']
            else:
                init_contact_mat = None

        self.m_g = edge_i
        if self.has_contact:
            self.m_cgs = init_contact_mat
        self.m_idx = [[-1]]

    def _cal_multi_mesh(self, fields):
        if not self.has_contact:
            if self.consist_mesh:
                mmfile = os.path.join(self.data_dir, 'mmesh_layer_' + str(self.layer_num) + '.dat')
            else:
                mmfile = os.path.join(self.data_dir, str(self.instance_id) + '_mmesh_layer_' + str(self.layer_num) + '.dat')
            mmexist = os.path.isfile(mmfile)
            if self.recal_mesh or not mmexist:
                if self.mesh_type == MeshType.Triangle:
                    edge_i = triangles_to_edges(self.cells)
                if self.mesh_type == MeshType.Tetrahedron:
                    edge_i = tetras_to_edges(self.cells)
                if self.mesh_type == MeshType.Quad:
                    edge_i = quads_to_edges(self.cells)
                if self.mesh_type == MeshType.Line:
                    edge_i = lines_to_edges(self.cells)
                if self.mesh_type == MeshType.Flat:
                    edge_i = self.cells
                m_gs, m_ids = generate_multi_layer_stride(edge_i,
                                                          self.layer_num,
                                                          seed_heuristic=self.seed_heuristic,
                                                          n=fields['mesh_pos'].shape[-2],
                                                          pos_mesh=fields["mesh_pos"][0].clone().detach().numpy())
                m_mesh = {'m_gs': m_gs, 'm_ids': m_ids}
                pickle.dump(m_mesh, open(mmfile, 'wb'))
            else:
                m_mesh = pickle.load(open(mmfile, 'rb'))
                m_gs, m_ids = m_mesh['m_gs'], m_mesh['m_ids']
            self.m_g = m_gs
            self.m_idx = m_ids
        else:
            self.contact_radius = self.meta['collision_radius']
            w_pos = fields['world_pos']
            num = w_pos.shape[-2]
            mmfile = os.path.join(self.data_dir, str(self.instance_id) + '_mcmesh_layer_' + str(self.layer_num) + '.dat')
            mmexist = os.path.isfile(mmfile)
            if self.recal_mesh or not mmexist:
                if self.mesh_type == MeshType.Triangle:
                    edge_i = triangles_to_edges(self.cells)
                if self.mesh_type == MeshType.Tetrahedron:
                    edge_i = tetras_to_edges(self.cells)
                # NOTE before removing and creating more clusters, using the origin adj to calculate contact pairs, this is faster for many clusters
                if self.has_self_contact:
                    pass
                else:
                    init_contact_mat = contact_edge_no_self(w_pos[:-1], edge_i, self.contact_radius)
                    # NOTE remove the connection between dirichlet nodes
                    edge_i = _remove_invalid_connection(edge_i, fields['node_type'][0], self.dirichelet_markers)
                # print(edge_i)
                m_gs, m_ids = generate_multi_layer_stride(edge_i,
                                                          self.layer_num,
                                                          seed_heuristic=self.seed_heuristic,
                                                          n=fields['mesh_pos'].shape[-2],
                                                          pos_mesh=fields["mesh_pos"][0].clone().detach().numpy())
                # enhance the contact adj
                m_cgs = []
                for it in range(self.len()):
                    # TODO broadcast among time
                    temp_wpos = w_pos[it, :, :]
                    if self.has_self_contact:
                        m_cg = multi_layer_contact_edge(m_gs, m_ids, temp_wpos, self.contact_radius, self_contact=self.has_self_contact)
                    else:
                        m_cg = multi_layer_contact_edge(m_gs, m_ids, temp_wpos, self.contact_radius, self_contact=self.has_self_contact, init_contact_g=init_contact_mat[it])
                    m_cgs.append(m_cg)
                # convert mgs to flat edge list
                m_mesh = {'m_gs': m_gs, 'm_ids': m_ids, 'm_cgs': m_cgs}
                pickle.dump(m_mesh, open(mmfile, 'wb'))

            else:
                m_mesh = pickle.load(open(mmfile, 'rb'))
                m_gs, m_ids, m_cgs = m_mesh['m_gs'], m_mesh['m_ids'], m_mesh['m_cgs']
            self.m_g = m_gs
            self.m_idx = m_ids
            self.m_cgs = m_cgs

        self.recal_mesh = False


class MeshCylinderDataset(MeshGeneralDataset):
    def __init__(self, root, instance_id, layer_num, stride, mode='train', noise_shuffle=False, noise_level=None, noise_gamma=1.0, recal_mesh=False, consist_mesh=False):
        in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list = ['velocity', 'mesh_pos'], ['velocity'], ['velocity']
        super().__init__(root,
                         in_normal_feature_list,
                         out_normal_feature_list,
                         roll_normal_feature_list,
                         instance_id,
                         layer_num,
                         stride,
                         mode,
                         noise_shuffle,
                         noise_level,
                         noise_gamma,
                         recal_mesh,
                         consist_mesh,
                         mesh_type=MeshType.Triangle,
                         has_contact=False,
                         has_self_contact=False,
                         dirichelet_markers=[1, 2, 3, 4],
                         seed_heuristic=SeedingHeuristic.MinAve)

    def _preprocess(self, fields):
        # noise shuffle
        # in: vel, type
        # out: d_vel
        node_info_inp = fields["velocity"][:-1].clone()
        node_info_tar = fields["velocity"][1:].clone()
        # enhance by noise level
        if self.noise_shuffle:
            # collect special nodes
            node_type = fields["node_type"][:-1]
            preset_node = ((node_type != 0) * (node_type != 5)).bool()
            no_noise_node = preset_node
            # collect special nodes
            noise_base = torch.ones_like(node_info_tar)
            noise_base[:, :] = self.noise_level
            noise = torch.normal(0.0, noise_base)
            # for dirichelet nodes, the noise is zero
            noise = torch.where(no_noise_node, torch.zeros_like(noise), noise)
            node_info_inp += noise
            node_info_tar += (1.0 - self.noise_gamma) * noise
        node_info_inp = torch.cat((node_info_inp, fields["mesh_pos"][:-1], fields["node_type"][:-1]), dim=-1)
        return node_info_inp, node_info_tar


class MeshAirfoilDataset(MeshGeneralDataset):
    def __init__(self, root, instance_id, layer_num, stride, mode='train', noise_shuffle=False, noise_level=None, noise_gamma=1.0, recal_mesh=False, consist_mesh=True):
        in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list, = ['velocity', 'density', 'mesh_pos'], ['velocity', 'density', 'pressure'], ['velocity', 'density']
        super().__init__(root,
                         in_normal_feature_list,
                         out_normal_feature_list,
                         roll_normal_feature_list,
                         instance_id,
                         layer_num,
                         stride,
                         mode,
                         noise_shuffle,
                         noise_level,
                         noise_gamma,
                         recal_mesh,
                         consist_mesh,
                         mesh_type=MeshType.Triangle,
                         has_contact=False,
                         has_self_contact=False,
                         dirichelet_markers=[1, 2, 3, 4],
                         seed_heuristic=SeedingHeuristic.MinAve)

    def _preprocess(self, fields):
        # noise shuffle
        # in: vel, density, pos, type
        # out: d_vel, d_density, pressure
        node_info_inp = torch.cat((fields["velocity"][:-1], fields["density"][:-1]), dim=-1)
        node_info_tar = torch.cat((fields["velocity"][1:], fields["density"][1:], fields["pressure"][1:]), dim=-1)
        # enhance by noise level
        if self.noise_shuffle:
            # collect special nodes
            node_type = fields["node_type"][:-1]
            preset_node = (node_type != 0).bool()
            no_noise_node = preset_node
            # collect special nodes
            noise_base = node_info_tar.new_ones((node_info_tar.shape[0], node_info_tar.shape[1], len(self.noise_level)), dtype=float)
            noise_base[:, :] = self.noise_level
            noise = torch.normal(0.0, noise_base)
            # for dirichelet nodes, the noise is zero
            noise = torch.where(no_noise_node, torch.zeros_like(noise), noise)
            node_info_inp += noise
            node_info_tar[..., :-1] += (1.0 - self.noise_gamma) * noise
        node_info_inp = torch.cat((node_info_inp, fields["mesh_pos"][:-1], fields["node_type"][:-1]), dim=-1)
        return node_info_inp, node_info_tar

    def suggested_pen_coef(self):
        pen_coef = super().suggested_pen_coef()
        # use KPa
        pen_coef[..., -1] /= 1e6
        return pen_coef


class MeshPlateDataset(MeshGeneralDataset):
    def __init__(self, root, instance_id, layer_num, stride, mode='train', noise_shuffle=False, noise_level=None, noise_gamma=1.0, recal_mesh=False, consist_mesh=False):
        in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list = ['world_pos', 'mesh_pos'], ['world_pos'], ['world_pos']
        super().__init__(root,
                         in_normal_feature_list,
                         out_normal_feature_list,
                         roll_normal_feature_list,
                         instance_id,
                         layer_num,
                         stride,
                         mode,
                         noise_shuffle,
                         noise_level,
                         noise_gamma,
                         recal_mesh,
                         consist_mesh,
                         mesh_type=MeshType.Tetrahedron,
                         has_contact=True,
                         has_self_contact=False,
                         dirichelet_markers=[1, 3])

    def _preprocess(self, fields):
        # noise shuffle
        # in: x,X,vel(for scripted nodes),type
        # out: d_x
        # noise: x
        node_info_inp = fields["world_pos"][:-1].clone()
        node_info_tar = fields["world_pos"][1:].clone()
        # enhance by noise level
        if self.noise_shuffle:
            # collect special nodes
            node_type = fields["node_type"][:-1]
            no_noise_node = ((node_type == 1) + (node_type == 3)).bool()
            noise_base = torch.ones_like(node_info_tar)
            noise_base[:, :] = self.noise_level
            noise = torch.normal(0.0, noise_base)
            # for dirichelet nodes, the noise is zero
            noise = torch.where(no_noise_node, torch.zeros_like(noise), noise)
            node_info_inp += noise
            node_info_tar += (1.0 - self.noise_gamma) * noise

        node_info_inp = torch.cat((node_info_inp, fields["mesh_pos"][:-1], fields["node_type"][:-1]), dim=-1)
        return node_info_inp, node_info_tar

    def get(self, idx):
        # idx in time seq (enhanced by noise shuffle)
        # also return the midx and mgs, for combining
        data = Data(x=self.in_feature[idx], y=self.tar_feature[idx])
        data.m_idx = self.m_idx  # const across time
        data.m_g = self.m_g  # const across time
        data.m_cg = self.m_cgs[idx]  # vary with time
        return data


class FontDataset(MeshGeneralDataset):
    def __init__(self, root, instance_id, layer_num, stride, mode='train', noise_shuffle=False, noise_level=None, noise_gamma=1.0, recal_mesh=False, consist_mesh=False):
        in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list = ['world_pos', 'mesh_pos'], ['world_pos'], ['world_pos']
        super().__init__(root,
                         in_normal_feature_list,
                         out_normal_feature_list,
                         roll_normal_feature_list,
                         instance_id,
                         layer_num,
                         stride,
                         mode,
                         noise_shuffle,
                         noise_level,
                         noise_gamma,
                         recal_mesh,
                         consist_mesh,
                         mesh_type=MeshType.Triangle,
                         has_contact=True,
                         has_self_contact=True,
                         seed_heuristic=SeedingHeuristic.NearCenter)

    def _preprocess(self, fields):
        # noise shuffle
        # in: x,X,type
        # out: d_x
        # noise: x
        node_info_inp = fields["world_pos"][:-1].clone()
        node_info_tar = fields["world_pos"][1:].clone()
        # enhance by noise level
        if self.noise_shuffle:
            # collect special nodes
            node_type = fields["node_type"][:-1]
            no_noise_node = (node_type != 0).bool()
            noise_base = torch.ones_like(node_info_tar)
            noise_base[:, :] = self.noise_level
            noise = torch.normal(0.0, noise_base)
            # for dirichelet nodes, the noise is zero
            noise = torch.where(no_noise_node, torch.zeros_like(noise), noise)
            node_info_inp += noise
            node_info_tar += (1.0 - self.noise_gamma) * noise
        node_info_inp = torch.cat((node_info_inp, fields["mesh_pos"][:-1], fields["node_type"][:-1]), dim=-1)
        return node_info_inp, node_info_tar

    def get(self, idx):
        # idx in time seq (enhanced by noise shuffle)
        # also return the midx and mgs, for combining
        data = Data(x=self.in_feature[idx], y=self.tar_feature[idx])
        data.m_idx = self.m_idx  # const across time
        data.m_g = self.m_g  # const across time
        data.m_cg = self.m_cgs[idx]  # vary with time
        return data


class HEAT1DDataset(MeshGeneralDataset):
    def __init__(self, root, instance_id, layer_num, stride, mode='train', noise_shuffle=False, noise_level=None, noise_gamma=1.0, recal_mesh=False, consist_mesh=False):
        in_normal_feature_list, out_normal_feature_list, roll_normal_feature_list = ['T', 'grad', 'mesh_pos'], ['T'], ['T'],
        super().__init__(root,
                         in_normal_feature_list,
                         out_normal_feature_list,
                         roll_normal_feature_list,
                         instance_id,
                         layer_num,
                         stride,
                         mode,
                         noise_shuffle,
                         noise_level,
                         noise_gamma,
                         recal_mesh,
                         consist_mesh,
                         mesh_type=MeshType.Line,
                         has_contact=False,
                         has_self_contact=False,
                         dirichelet_markers=[],
                         seed_heuristic=SeedingHeuristic.MinAve)

    def _preprocess(self, fields):
        # noise shuffle
        # raw in: T,grad,X,type
        # noise: none
        # preprocess: set all int(0) T and grad to zero
        T_inp = fields["T"][:-1].clone()
        grad_inp = fields["grad"][:-1].clone()
        node_info_tar = fields["T"][1:].clone()
        node_type = fields["node_type"][:-1].unsqueeze(-1)
        int_node = (node_type == 0).bool()
        # print(int_node.shape)
        # print(T_inp.shape)
        T_inp = torch.where(int_node, torch.zeros_like(T_inp), T_inp)
        grad_inp = torch.where(int_node, torch.zeros_like(grad_inp), grad_inp)
        node_info_inp = torch.cat((T_inp, grad_inp, fields["mesh_pos"][:-1], node_type), dim=-1)
        return node_info_inp, node_info_tar
