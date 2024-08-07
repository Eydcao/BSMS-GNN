import os
import glob
import pickle
import numpy as np
import h5py
import torch
import torchdata.datapipes as dp
from graph_wrappers import BistrideMultiLayerGraph
from utils import to_flat_edge
from torch_geometric.data import Data


class singleTrajReader:
    def __init__(self, cfg, file_path, mode):
        """
        Initialize the singleTrajReader object.

        Parameters
        ----------
        cfg : OmegaConf
            Configuration object containing all necessary parameters.
        file_path : str
            Path to the specific transient sequence file.
        mode : str
            Mode of the dataset, e.g., 'train', 'val', or 'test'.
        """
        self.file_path = file_path
        self.mode = mode
        self.data_dir = os.path.dirname(file_path)
        self.unet_depth = cfg.unet_depth
        self.consist_mesh = cfg.consist_mesh
        self.mesh_type = cfg.mesh_type
        fields = dict()

        with h5py.File(file_path, "r") as f:
            for name in cfg.field_names:
                if name == "cells":
                    fields[name] = np.array(f[name])
                    self.cells = fields[name][0]
                else:
                    fields[name] = torch.tensor(np.array(f[name]), dtype=torch.float)

        # Calculate length of dataset, -1 because the input cannot be the last T_idx, otherwise no target
        self.L = fields["mesh_pos"].shape[0] - 1

        # Calculate multi-level mesh
        self._cal_multi_mesh(fields["mesh_pos"][0].clone().numpy())

        self.fields = fields

    def __len__(self):
        """
        Get the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return self.L

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        tuple
            Tuple of input features and target features.
        """
        return {k: v[idx] for k, v in self.fields.items()}, {k: v[idx + 1] for k, v in self.fields.items()}

    def get_full_frame(self):
        """
        Get the full frame of the dataset.

        Returns
        -------
        dict
            Dictionary of all fields in the dataset.
        """
        return {k: v for k, v in self.fields.items()}

    def _cal_multi_mesh(self, mesh_pos):
        """
        Calculate the multi-layer mesh.

        Parameters
        ----------
        mesh_pos : np.ndarray
            The positions of the mesh nodes.
        """
        mmfile = os.path.join(
            self.data_dir,
            f"{'' if self.consist_mesh else os.path.basename(self.file_path) + '_'}mmesh_layer_{self.unet_depth}.dat",
        )
        if not os.path.isfile(mmfile):
            edge_i = to_flat_edge(self.cells, self.mesh_type)
            num_nodes = mesh_pos.shape[0]
            m_graph = BistrideMultiLayerGraph(edge_i, self.unet_depth, num_nodes, mesh_pos)
            _, m_gs, m_ids = m_graph.get_multi_layer_graphs()
            m_gs = [torch.tensor(g, dtype=torch.long) for g in m_gs]
            m_ids = [torch.tensor(ids, dtype=torch.long) for ids in m_ids]
            # check m_gs and m_ids
            print(("mgs shapes"))
            for g in m_gs:
                print(g.shape)
            print(("m_ids shapes"))
            for i in m_ids:
                print(i.shape)
            m_mesh = {"m_gs": m_gs, "m_ids": m_ids}
            with open(mmfile, "wb") as f:
                pickle.dump(m_mesh, f)
        else:
            with open(mmfile, "rb") as f:
                m_mesh = pickle.load(f)
            m_gs, m_ids = m_mesh["m_gs"], m_mesh["m_ids"]

        self.m_gs = m_gs
        self.m_ids = m_ids


class TrajectoryDataPipe(dp.iter.IterDataPipe):
    def __init__(self, cfg, num_workers, base_seed, mode):
        """
        Initialize the TrajectoryDataPipe object.

        Parameters
        ----------
        cfg : OmegaConf
            Configuration object containing all necessary parameters.
        num_workers : int
            Number of worker processes for data loading.
        base_seed : int
            Base seed for random number generators.
        mode : str
            Mode of the dataset, e.g., 'train', 'val', or 'test'.
        """
        self.cfg = cfg
        self.num_workers = num_workers
        self.base_seed = base_seed
        self.train = mode == "train"
        self.rollout = mode == "rollout"
        # NOTE there is no subdir "rollout" in the data directory, change it to "test"
        if self.rollout:
            mode = "test"
        self.mode = mode
        self.data_dir = os.path.join(cfg.root, cfg.tf_dataset_name, mode)
        self.file_list = self._get_file_list()
        self.rng = None
        self.tc_rng = None
        self._init_rng()

    def _get_file_list(self):
        """
        Get the list of files in the data directory.

        Returns
        -------
        list
            List of file paths.
        """
        return glob.glob(os.path.join(self.data_dir, "*.h5"))

    def _init_rng(self):
        """
        Initialize different random generators for each worker.
        """
        worker_id, _ = self._get_worker_id_and_info()
        train_seed = np.random.randint(1000)
        seed_list = [train_seed, worker_id, self.base_seed]
        seed = self._hash(seed_list, 1000)
        self.rng = np.random.default_rng(seed)
        self.tc_rng = torch.Generator()
        self.tc_rng.manual_seed(seed)

    def _get_worker_id_and_info(self):
        worker_info = torch.utils.data.get_worker_info()
        return 0 if worker_info is None else worker_info.id, worker_info

    def _hash(self, list, base):
        """
        Compute a hash for a list using a base value.

        Parameters
        ----------
        list : list
            List to hash.
        base : int
            Base value for hashing.

        Returns
        -------
        int
            Computed hash value.
        """
        hash = 0
        for i in range(len(list)):
            hash += list[i] * (base**i)
        return hash

    def _get_nested_paths(self):
        """
        Split data paths based on the number of workers.

        Returns
        -------
        list
            List of nested paths for each worker.
        """
        if self.num_workers <= 1:
            return [self.file_list]
        else:
            return np.array_split(self.file_list, self.num_workers)

    def _get_mask(self, node_type):
        """
        Get the mask for nodes. 1 represents the node is valid, 0 represents the node is NA for loss calculation.
        Usually, 0 masks are either Dirichelet nodes or outgoing bc.

        Parameters
        ----------
        node_type: torch.Tensor of shape (N,)

        Returns
        -------
        tuple
        node_mask: torch.Tensor of shape (N, 1)
        """
        # NOTE this function must be defined in the child dataset class
        return NotImplementedError("_get_mask must be defined in the child dataset class.")

    def _proc_data(self, data, rng, tc_rng):
        """
        Process the fields and inject noise if necessary.

        Parameters
        ----------
        data : tuple
            Tuple of input and target features (dict of in, dict of tar).
        rng : np.random.Generator
            Random number generator for numpy.
        tc_rng : torch.Generator
            Random number generator for torch.

        Returns
        -------
        tuple
            Tuple of processed input features and target features.
        """
        fields_inp, fields_tar = data

        # (T, N, C)
        keys_out = list(self.cfg.output_field_names)
        keys_in = [*keys_out, "mesh_pos", "node_type"]
        node_in_list = []
        node_tar_list = []
        for key in keys_out:
            node_tar_list.append(fields_tar[key])
        for key in keys_in:
            node_in_list.append(fields_inp[key])
        # to torch
        node_info_inp = torch.cat(node_in_list, dim=-1)
        node_info_tar = torch.cat(node_tar_list, dim=-1)

        # generate node mask
        node_mask = self._get_mask(fields_inp["node_type"])

        if self.train:
            # collect special nodes
            no_noise_node = (node_mask == 0).bool()

            # (T, N, c=len(self.cfg.noise_level))
            noise_base = node_info_tar.new_ones(node_info_tar.shape, dtype=float)
            noise_base[:, :] = torch.tensor(np.array(list(self.cfg.noise_level)))
            # use tc_rng to generate guassian noise, mean=0 , std=noise_base
            noise = torch.normal(mean=node_info_tar.new_zeros(node_info_tar.shape), std=noise_base, generator=tc_rng)

            # for Dirichlet nodes, the noise is zero
            noise = torch.where(no_noise_node, torch.zeros_like(noise), noise)
            c_len = noise.shape[-1]
            # -1 is the type; no need to add noise
            node_info_inp[:, :c_len] += noise
            node_info_tar += (1.0 - self.cfg.noise_gamma) * noise

        return node_info_inp, node_info_tar, node_mask

    def __iter__(self):
        """
        Iterate over the dataset, yielding data samples.

        Yields
        ------
        tuple
            Tuple of input features and target features.
        """
        worker_id, _ = self._get_worker_id_and_info()
        worker_paths = self._get_nested_paths()[worker_id]
        if self.rng is not None:
            self.rng.shuffle(worker_paths)

        for file_path in worker_paths:
            traj_reader = singleTrajReader(self.cfg, file_path, self.mode)
            t_len = len(traj_reader)
            t_ids = np.arange(t_len)
            if not self.rollout:
                if self.rng is not None:
                    self.rng.shuffle(t_ids)

                for ti in t_ids:
                    data = traj_reader[ti]
                    node_info_inp, node_info_tar, node_mask = self._proc_data(data, self.rng, self.tc_rng)

                    if self.cfg.consist_mesh:
                        yield node_info_inp, node_info_tar, node_mask, traj_reader.m_gs, traj_reader.m_ids
                    else:
                        m_gs = []
                        # NOTE use the 'face' attribute so the pyg will shift the indices automatically
                        # Level 0
                        level_0_data = Data(
                            x=node_info_inp,
                            y=node_info_tar,
                            mask=node_mask,
                            edge_index=traj_reader.m_gs[0],
                            face=traj_reader.m_ids[0],
                            num_nodes=node_info_inp.shape[-2],
                        )
                        m_gs.append(level_0_data)

                        # Levels 1 and above
                        for i in range(1, len(traj_reader.m_gs)):
                            if not i == len(traj_reader.m_gs) - 1:
                                level_data = Data(
                                    face=traj_reader.m_ids[i],
                                    edge_index=traj_reader.m_gs[i],
                                    num_nodes=traj_reader.m_ids[i - 1].shape[0],
                                )
                            else:
                                level_data = Data(
                                    # last layer no need to pool
                                    edge_index=traj_reader.m_gs[i],
                                    num_nodes=traj_reader.m_ids[i - 1].shape[0],
                                )
                            m_gs.append(level_data)

                        yield m_gs
            else:
                data = traj_reader[t_ids]
                # (T-1, N, C), (T-1, N, C), (T-1, N, 1)
                node_info_inp, node_info_tar, node_mask = self._proc_data(data, self.rng, self.tc_rng)
                # since in rollout mesh is always consistent for each traj
                yield node_info_inp, node_info_tar, node_mask, traj_reader.m_gs, traj_reader.m_ids
