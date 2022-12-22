import numpy as np
from sklearn import neighbors
from helpers_convert import _flat_edge_to_adj_list, _flat_edge_to_adj_mat, _adj_mat_to_flat_edge
from helpers_BFS import _find_clusters


def _compute_connectivity_ij(positionsi, positionj, radius):
    # find connections from cluster i to cluster j
    tree = neighbors.KDTree(positionj)
    r_list = tree.query_radius(positionsi, r=radius)
    num_nodes = len(positionsi)
    s = np.repeat(range(num_nodes), [len(a) for a in r_list])
    r = np.concatenate(r_list, axis=0)

    return s, r


def contact_edge_no_self(pos, original_flat_edge, radius):
    # find clusters, get pos of each cluster, for each pair between cluster pos(2 nested loop, upper trianglized), call contact_between_cluster_pos
    batched = (len(pos.shape) == 3)
    adj_list = _flat_edge_to_adj_list(original_flat_edge)
    # cal clustered pos
    clusters = _find_clusters(adj_list)
    if batched:
        pos_cs = [pos[:, c] for c in clusters]
    else:
        pos_cs = [pos[c] for c in clusters]

    if len(clusters) == 0:
        pass
    else:
        if batched:
            res = []
            for t in range(pos.shape[0]):
                cont_s = []
                cont_r = []
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        s, r = _compute_connectivity_ij(pos_cs[i][t], pos_cs[j][t], radius)
                        s = [clusters[i][e] for e in s]
                        r = [clusters[j][e] for e in r]
                        cont_s += s + r
                        cont_r += r + s
                flat_edge = np.array([cont_s, cont_r])
                res.append(flat_edge)
            return res
        else:
            cont_s = []
            cont_r = []
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    s, r = _compute_connectivity_ij(pos_cs[i], pos_cs[j], radius)
                    s = [clusters[i][e] for e in s]
                    r = [clusters[j][e] for e in r]
                    cont_s += s + r
                    cont_r += r + s
            flat_edge = np.array([cont_s, cont_r])
            return flat_edge


def _remove_existing_edges(g_c, g):
    gc_mat = _flat_edge_to_adj_mat(g_c).toarray()
    g_mat = _flat_edge_to_adj_mat(g).toarray()
    gc_mat = (gc_mat * (1 - g_mat)).astype(bool).astype(float)
    g_c = _adj_mat_to_flat_edge(gc_mat)
    return g_c


def contact_edge(pos, original_flat_edge, radius):
    batched = (len(pos.shape) == 3)
    if batched:
        res = []
        for i in range(pos.shape[0]):
            s_pos = pos[i]
            s, r = _compute_connectivity_ij(s_pos, s_pos, radius)
            g = np.array([s, r])
            g = _remove_existing_edges(g, original_flat_edge)
            res.append(g)
        return res
    else:
        s, r = _compute_connectivity_ij(pos, pos, radius)
        g = np.array([s, r])
        g = _remove_existing_edges(g, original_flat_edge)
        return g
