import numpy as np

_INF = 1 + 1e10


def _BFS_dist(adj_list, n_nodes, seed, mask=None):
    # mask: meaning only search within the subset indicated by it, any outside nodes are not reachable
    #       can be achieved by marking outside nodes as visited, dist to inf
    res = np.ones(n_nodes) * _INF
    vistied = [False for _ in range(n_nodes)]
    if isinstance(seed, list):
        for s in seed:
            res[s] = 0
            vistied[s] = True
        frontier = seed
    else:
        res[seed] = 0
        vistied[seed] = True
        frontier = [seed]

    if isinstance(mask, list):
        for i, m in enumerate(mask):
            if m != True:
                res[i] = _INF
                vistied[i] = True

    depth = 0
    track = [frontier]
    while frontier:
        this_level = frontier
        depth += 1
        frontier = []
        while this_level:
            f = this_level.pop(0)
            for n in adj_list[f]:
                if not vistied[n]:
                    vistied[n] = True
                    frontier.append(n)
                    res[n] = depth
        # record each level
        track.append(frontier)

    return res, track


def _BFS_dist_all(adj_list, n_nodes):
    res = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        res[i], _ = _BFS_dist(adj_list, n_nodes, i)
    return res


def _find_clusters(adj_list, mask=None):
    n_nodes = len(adj_list)
    if isinstance(mask, list):
        remaining_nodes = []
        for i, m in enumerate(mask):
            if m == True:
                remaining_nodes.append(i)
    else:
        remaining_nodes = list(range(n_nodes))
    cluster = []
    while remaining_nodes:
        if len(remaining_nodes) > 1:
            seed = remaining_nodes[0]
            dist, _ = _BFS_dist(adj_list, n_nodes, seed, mask)
            tmp = []
            new_remaining = []
            for n in remaining_nodes:
                if dist[n] != _INF:
                    tmp.append(n)
                else:
                    new_remaining.append(n)
            cluster.append(tmp)
            remaining_nodes = new_remaining
        else:
            cluster.append([remaining_nodes[0]])
            break

    return cluster
