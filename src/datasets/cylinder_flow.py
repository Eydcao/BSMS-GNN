from .base import TrajectoryDataPipe
import torch


class cylinderDataPipe(TrajectoryDataPipe):
    def __init__(self, cfg, num_workers, base_seed, mode):
        super().__init__(cfg, num_workers, base_seed, mode)

    def _get_mask(self, node_type):
        """
        Get the mask for nodes. 1 represents the node is valid, 0 represents the node is NA for loss calculation.
        Usually, 0 masks are either Dirichelet nodes or outgoing bc.

        Parameters
        ----------
        node_type: torch.Tensor of shape (N, 1)

        Returns
        -------
        tuple
        node_mask: torch.Tensor of shape (N, 1)
        """
        # For cylinder, all nodes with 0 (interior) or 5 (outflow) are valid
        mask = torch.logical_or(node_type == 0, node_type == 5).float()
        return mask
