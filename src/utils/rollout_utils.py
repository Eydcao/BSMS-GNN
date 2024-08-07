import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def restore_model(model, ckpt_path):
    # Model restoration from the last checkpoint in store_dir
    model = model.module if hasattr(model, "module") else model
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("restored params from {}".format(ckpt_path))


@torch.no_grad()
def rollout_one_traj(trainer, IC, results, node_mask, m_gs, m_ids, cfg):
    """
    Rollout one trajectory.

    Parameters
    ----------
    trainer : Trainer
        Trainer object.
    IC : torch.Tensor
        Initial condition. (1, N, C+dim+1)
    results : torch.Tensor
        Results of prediction, initially a zero dummy, (T-1, N, C).
    node_mask : torch.Tensor, (1, N, 1)
        Node mask.
    m_ids : torch.Tensor
        Multi-layer IDs.
    m_gs : list
        Multi-layer graphs.
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    torch.Tensor
        Results of the current rollout.
    """

    T_len = results.shape[0]
    current_input = IC.clone()
    # get the mesh_pos and node_type from the initial condition, ie the last dim+1 channels
    # (1, N, pos_dim+1)
    mesh_post_type = current_input[..., results.shape[-1] :].clone()

    for ti in range(T_len):
        data = (current_input, current_input.new_zeros(current_input.shape), node_mask, m_gs, m_ids)
        # (1, N, C)
        pred = trainer.model(data, consistent_mesh=True, warmup=False)
        # assign to results container
        results[ti] = pred[0]

        # create next time input
        # (1, N, C)
        current_input = pred
        # cat the mesh_pos and node_type to the end
        # (1, N, C+dim+1)
        current_input = torch.cat([current_input, mesh_post_type], dim=-1)
        # assign any unmasked node to IC, as they are fixed
        current_input = torch.where(node_mask == 0, IC, current_input)

    return results
