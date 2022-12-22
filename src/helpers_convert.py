import numpy as np
import scipy


def _adj_list_to_mat(adj_list):
    n_node = len(adj_list)
    adj_mat = np.zeros((n_node, n_node))
    for i in range(n_node):
        adj_mat[i, i] = 1
        for n in adj_list[i]:
            adj_mat[i, n] = 1
    return adj_mat


def _adj_mat_to_list(adj_mat):
    n_node = adj_mat.shape[0]
    flat_e = _adj_mat_to_flat_edge(adj_mat)
    adj_list = _flat_edge_to_adj_list(flat_e, n_node)
    return adj_list


def _flat_edge_to_adj_mat(edge_list, n=None):
    if n is None:
        sender_node = set(edge_list[0])
        receiv_node = set(edge_list[1])
        all_node = list(sender_node.union(receiv_node))
        n = max(all_node) + 1
    adj_mat = scipy.sparse.coo_array((np.ones_like(edge_list[0]), (edge_list[0], edge_list[1])), shape=(n, n))

    return adj_mat


def _flat_edge_to_adj_list(edge_list, n=None):
    if n == None:
        sender_node = set(edge_list[0])
        receiv_node = set(edge_list[1])
        all_node = list(sender_node.union(receiv_node))
        n = max(all_node) + 1
    adj_list = [[] for _ in range(n)]
    for i in range(len(edge_list[0])):
        adj_list[edge_list[0, i]].append(edge_list[1, i])

    return adj_list


def _adj_list_to_flat_edge(adj_list):
    edge_list = []
    for i in range(len(adj_list)):
        for n in adj_list[i]:
            edge_list.append([i, n])
    return np.array(edge_list).transpose()


def _adj_mat_to_flat_edge(adj_mat):
    if isinstance(adj_mat, np.ndarray):
        s, r = np.where(adj_mat.astype(bool))
    elif isinstance(adj_mat, scipy.sparse.coo_array):
        s, r = adj_mat.row, adj_mat.col
        dat = adj_mat.data
        valid = np.where(dat.astype(bool))[0]
        s, r = s[valid], r[valid]
    elif isinstance(adj_mat, scipy.sparse.csr_matrix):
        adj_mat = scipy.sparse.coo_array(adj_mat)
        s, r = adj_mat.row, adj_mat.col
        dat = adj_mat.data
        valid = np.where(dat.astype(bool))[0]
        s, r = s[valid], r[valid]
    else:
        print('tobe implemented _adj_mat_to_flat_edge')
        exit(1)
    return np.array([s, r])
