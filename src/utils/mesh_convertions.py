import torch


def triangles_to_edges(cells):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    t_cell = torch.tensor(cells)
    edge_index = torch.cat(
        (t_cell[:, :2], t_cell[:, 1:3], torch.cat((t_cell[:, 2].unsqueeze(1), t_cell[:, 0].unsqueeze(1)), -1)), 0
    )
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single torch long tensor
    r, _ = torch.min(edge_index, 1, keepdim=True)
    s, _ = torch.max(edge_index, 1, keepdim=True)
    packed_edges = torch.cat((s, r), 1).type(torch.long)
    # remove duplicates and unpack
    unique_edges = torch.unique(packed_edges, dim=0)
    s, r = unique_edges[:, 0], unique_edges[:, 1]
    # create two-way connectivity
    return torch.stack((torch.cat((s, r), 0), torch.cat((r, s), 0))).numpy()


def tetras_to_edges(cells):
    """Computes mesh edges from tetrahedrons."""
    # collect edges from triangles
    t_cell = torch.tensor(cells)
    edge_index = torch.cat(
        (
            t_cell[:, :2],
            t_cell[:, 1:3],
            t_cell[:, 2:4],
            torch.cat((t_cell[:, 3].unsqueeze(1), t_cell[:, 0].unsqueeze(1)), -1),
            torch.cat((t_cell[:, 0].unsqueeze(1), t_cell[:, 2].unsqueeze(1)), -1),
            torch.cat((t_cell[:, 1].unsqueeze(1), t_cell[:, 3].unsqueeze(1)), -1),
        ),
        0,
    )
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single torch long tensor
    r, _ = torch.min(edge_index, 1, keepdim=True)
    s, _ = torch.max(edge_index, 1, keepdim=True)
    packed_edges = torch.cat((s, r), 1).type(torch.long)
    # remove duplicates and unpack
    unique_edges = torch.unique(packed_edges, dim=0)
    s, r = unique_edges[:, 0], unique_edges[:, 1]
    # create two-way connectivity
    return torch.stack((torch.cat((s, r), 0), torch.cat((r, s), 0))).numpy()


def quads_to_edges(cells):
    """Computes mesh edges from quads."""
    # collect edges from quads
    t_cell = torch.tensor(cells)
    edge_index = torch.cat(
        (
            t_cell[:, :2],
            t_cell[:, 1:3],
            t_cell[:, 2:4],
            torch.cat((t_cell[:, 3].unsqueeze(1), t_cell[:, 0].unsqueeze(1)), -1),
        ),
        0,
    )
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single torch long tensor
    r, _ = torch.min(edge_index, 1, keepdim=True)
    s, _ = torch.max(edge_index, 1, keepdim=True)
    packed_edges = torch.cat((s, r), 1).type(torch.long)
    # remove duplicates and unpack
    unique_edges = torch.unique(packed_edges, dim=0)
    s, r = unique_edges[:, 0], unique_edges[:, 1]
    # create two-way connectivity
    return torch.stack((torch.cat((s, r), 0), torch.cat((r, s), 0))).numpy()


def lines_to_edges(cells):
    """Computes mesh edges from lines."""
    # collect edges from quads
    t_cell = torch.tensor(cells)
    # print(t_cell)
    s, r = t_cell[0], t_cell[1]
    # create two-way connectivity
    return torch.stack((torch.cat((s, r), 0), torch.cat((r, s), 0))).numpy()


def to_flat_edge(mesh, mesh_type):
    if mesh_type == "tri":
        return triangles_to_edges(mesh)
    elif mesh_type == "tetra":
        return tetras_to_edges(mesh)
    elif mesh_type == "quad":
        return quads_to_edges(mesh)
    elif mesh_type == "line":
        return lines_to_edges(mesh)
    elif mesh_type == "flat":
        return mesh
    else:
        raise ValueError(f"Unsupported mesh type {mesh_type} in to_flat_edge.")
