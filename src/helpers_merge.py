import numpy as np
import copy


def multi_layered_offset(b_midx, b_num_node):
    # shape B, L, ~; B,L+1,2,~
    # create multi-layered offset
    # shape: Layer, Cum_sumed_offset(B)
    m_offset = [np.cumsum(b_num_node)]
    for l in range(len(b_midx[0])):
        len_this_layer = np.array([len(midx[l]) for midx in b_midx])
        offset_this_layer = np.cumsum(len_this_layer)
        m_offset.append(offset_this_layer)

    return m_offset


def merge_mids(b_midx, b_num_node):
    if not isinstance(b_midx[0], list):
        # for flat GMP, this is None
        return None
    else:
        m_offset = multi_layered_offset(b_midx, b_num_node)
        for i, midx in enumerate(b_midx):
            if i == 0:
                merged_midx = copy.deepcopy(midx)
            else:
                # NOTE add offset each layer
                # NOTE merge each layer
                for layer in range(len(merged_midx)):
                    offseted_idx = [id + m_offset[layer][i - 1] for id in midx[layer]]
                    merged_midx[layer] += offseted_idx

        return merged_midx


def merge_mgs(b_mgs, b_midx, b_num_node):
    m_offset = multi_layered_offset(b_midx, b_num_node)
    for i, mgs in enumerate(b_mgs):
        if i == 0:
            merged_mgs = mgs.copy()
        else:
            # NOTE add offset each layer
            # NOTE merge each layer
            for layer in range(len(merged_mgs)):
                offset_this_layer_this_g = m_offset[layer][i - 1]
                offseted_gs_this_layer = mgs[layer] + offset_this_layer_this_g
                merged_mgs[layer] = np.concatenate((merged_mgs[layer], offseted_gs_this_layer), axis=1)

    return merged_mgs


def multi_layered_offset_by_len(b_n, m_shift_n):
    # shape B, L, ~; B,L+1,2,~
    # create multi-layered offset
    # shape: Layer, Cum_sumed_offset(B)
    m_offset = []
    for offset in m_shift_n:
        len_this_layer = np.array([offset] * b_n)
        offset_this_layer = np.cumsum(len_this_layer)
        m_offset.append(offset_this_layer)

    return m_offset


def merge_mgs_by_len(b_mgs, m_shift_n):
    m_offset = multi_layered_offset_by_len(len(b_mgs), m_shift_n)
    for i, mgs in enumerate(b_mgs):
        if i == 0:
            merged_mgs = mgs.copy()
        else:
            # NOTE add offset each layer
            # NOTE merge each layer
            for layer in range(len(merged_mgs)):
                offset_this_layer_this_g = m_offset[layer][i - 1]
                offseted_gs_this_layer = mgs[layer] + offset_this_layer_this_g
                merged_mgs[layer] = np.concatenate((merged_mgs[layer], offseted_gs_this_layer), axis=1)

    return merged_mgs


def multi_layered_offset_by_unique_len(b_aggr_list):
    # shape B, L, ~; B,L+1,2,~
    # create multi-layered offset
    # shape: Layer, Cum_sumed_offset(B)
    m_offset = []
    for l in range(len(b_aggr_list[0])):
        len_this_layer = np.array([max(aggr_list[l]) + 1 for aggr_list in b_aggr_list])
        offset_this_layer = np.cumsum(len_this_layer)
        m_offset.append(offset_this_layer)

    return m_offset


def merge_aggr_lists(b_aggr_list):
    if not isinstance(b_aggr_list[0], list):
        # for flat GMP, this is None
        return None
    else:
        m_offset = multi_layered_offset_by_unique_len(b_aggr_list)
        for i, m_aggr_list in enumerate(b_aggr_list):
            if i == 0:
                merged_m_aggr_list = copy.deepcopy(m_aggr_list)
            else:
                # NOTE add offset each layer
                # NOTE merge each layer
                for layer in range(len(merged_m_aggr_list)):
                    offseted_idx = m_aggr_list[layer] + m_offset[layer][i - 1]
                    merged_m_aggr_list[layer] = np.concatenate((merged_m_aggr_list[layer], offseted_idx), axis=0)

        return merged_m_aggr_list


def merge_pos(b_m_pos):
    for i, m_pos in enumerate(b_m_pos):
        if i == 0:
            merged_m_pos = copy.deepcopy(m_pos)
        else:
            for layer in range(len(merged_m_pos)):
                merged_m_pos[layer] = np.concatenate((merged_m_pos[layer], m_pos[layer]), axis=0)

    return merged_m_pos