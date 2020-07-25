from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import coo_matrix
from sklearn.neighbors import KDTree
import scipy.sparse as sp
from settings import *
import gcn.embedding as gcn
import networkx as nx
import numpy as np
import pandas as pd
import math

"""
Calculating embeddings for sup_nodes and original nodes, as well as alignment accuracy.
"""


def ini_settings(args):
    num_buckets = args.num_buckets  # base of log for log scale
    if num_buckets == 1:
        num_buckets = None
    rept_settings = ParaSettings(max_layer=args.max_layer, alpha=args.alpha, k=args.k, num_buckets=num_buckets,
                                 normalize=True, gamma_struc=args.gamma_struc, gamma_attr=args.gamma_attr)
    return rept_settings


def get_embed_gcn(nx_g1, nx_g2, g1_nodes_num, g2_nodes_num, rep_settings, net_settings, g1_node_attrs=None,
                  g2_node_attrs=None):
    rep_g1 = Graph(nx.adjacency_matrix(nx_g1), all_nodes_num=g1_nodes_num)
    rep_g2 = Graph(nx.adjacency_matrix(nx_g2), all_nodes_num=g2_nodes_num)
    combine_net = nx.compose(nx_g1, nx_g2)
    combine_adj = nx.adjacency_matrix(combine_net)
    feature_matrix, fea_num = get_nodes_feature(rep_g1, rep_g2, rep_settings, g1_node_attrs, g2_node_attrs)

    # Run gcn model to get embeddings with a combined matrix of two networks
    print("learning representations...")
    emb_shape = [rep_g1.all_nodes_num + rep_g2.all_nodes_num, fea_num * 2]
    embeddings = gcn.get_embeddings(combine_adj, feature_matrix, emb_shape, net_settings)
    return embeddings


def get_nodes_feature(rep_g1, rep_g2, rep_settings, g1_node_attrs, g2_node_attrs):
    """
    Get structural features for all nodes in two graphs based on degree sequences of neighbors
    :return: Discontinuous node ids in graph are matched to 0 to max node num in feature_matrix
    """
    # Get k-hop neighbors of all nodes. And degree of each node in graph, index starts from 0 to maximum.
    rep_g1_k_hop_neighbors, rep_g1_node_degrees = get_k_hop_neighbs(rep_g1, rep_settings)
    rep_g2_k_hop_neighbors, rep_g2_node_degrees = get_k_hop_neighbs(rep_g2, rep_settings)
    max_degree = max(max(rep_g1_node_degrees), max(rep_g2_node_degrees))

    if rep_settings.num_buckets is None:  # 1 bin for every possible degree value
        num_features = max_degree + 1  # count from 0 to max degree, could change if bucketing degree sequences
    else:  # logarithmic binning with num_buckets as the base of logarithm (default: base 2)
        num_features = int(math.log(max_degree, rep_settings.num_buckets)) + 1

    total_nodes_num = rep_g1.all_nodes_num + rep_g2.all_nodes_num
    # Discontinuous node ids in two graphs are matched to 0 to max node num in feature_matrix
    feature_matrix = np.zeros((total_nodes_num, num_features))
    index = 0
    for n in range(rep_g1.all_nodes_num):
        for layer in rep_g1_k_hop_neighbors[n].keys():  # construct feature matrix one layer at a time
            if len(rep_g1_k_hop_neighbors[n][layer]) > 0:
                # degree sequence of node n at a hop layer
                deg_seq = get_degree_sequence(rep_g1_node_degrees, rep_settings.num_buckets,
                                              rep_g1_k_hop_neighbors[n][layer], max_degree)
                # add degree info from this degree sequence, weighted depending on layer and discount factor alpha
                feature_matrix[index] += [(rep_settings.alpha ** layer) * x for x in deg_seq]
        index += 1
    for n in range(rep_g2.all_nodes_num):
        for layer in rep_g2_k_hop_neighbors[n].keys():  # construct feature matrix one layer at a time
            if len(rep_g2_k_hop_neighbors[n][layer]) > 0:
                # degree sequence of node n at layer "layer"
                deg_seq = get_degree_sequence(rep_g2_node_degrees, rep_settings.num_buckets,
                                              rep_g2_k_hop_neighbors[n][layer], max_degree)
                feature_matrix[index] += [(rep_settings.alpha ** layer) * x for x in deg_seq]
        index += 1

    if g1_node_attrs is not None and g2_node_attrs is not None:
        node_attrs = np.concatenate((g1_node_attrs, g2_node_attrs), axis=0)
        feature_matrix = np.concatenate((feature_matrix, node_attrs), axis=1)
        num_features = feature_matrix.shape[1]

    return feature_matrix, num_features


def get_k_hop_neighbs(rep_g, rep_settings):
    """
    For each node, dictionary containing {node : {layer_num : {set of neighbors}}}.
    Discontinuous node ids in graph are matched to 0 to max node num in matrix
    """
    if rep_settings.max_layer is None:
        rep_settings.max_layer = rep_g.all_nodes_num  # prevent infinite loop

    # discontinuous node ids in graph are matched to 0 to max node num in matrix
    k_hop_neighbors_dict = {}
    all_neighbors_traversed = {}
    node_degrees = []

    # 0-hop neighbor of a node is itself
    for node_ind in range(rep_g.all_nodes_num):
        neighbors = np.nonzero(rep_g.G_adj[node_ind])[-1].tolist()  # column indices of non-zero elements
        node_degrees.append(len(neighbors))

        if len(neighbors) == 0:
            print("Warning: node %d is disconnected " + str(node_ind))
            k_hop_neighbors_dict[node_ind] = {0: {node_ind}, 1: set()}
        else:
            if type(neighbors[0]) is list:
                neighbors = neighbors[0]
            k_hop_neighbors_dict[node_ind] = {0: {node_ind}, 1: set(neighbors) - {node_ind}}
        # For each node, keep track of neighbors we've already seen
        all_neighbors_traversed[node_ind] = {node_ind}.union(k_hop_neighbors_dict[node_ind][1])

    current_layer = 2
    while True:
        if rep_settings.max_layer is not None and current_layer > rep_settings.max_layer:
            break
        reached_graph_diameter = True  # whether we've reached the graph diameter

        for i in range(rep_g.all_nodes_num):
            prev_hop_neighbors = k_hop_neighbors_dict[i][current_layer - 1]
            current_hop_neighbors = set()

            for n in prev_hop_neighbors:
                neighbors_of_n = k_hop_neighbors_dict[n][1]
                for node_id in neighbors_of_n:
                    current_hop_neighbors.add(node_id)
            # remove already seen nodes (k-hop neighbors reachable at shorter hop distance)
            current_hop_neighbors = current_hop_neighbors - all_neighbors_traversed[i]

            # Add neighbors at this hop to set of nodes we've already seen
            all_neighbors_traversed[i] = all_neighbors_traversed[i].union(current_hop_neighbors)

            # we've not reached the graph diameter in this round
            if len(current_hop_neighbors) > 0:
                reached_graph_diameter = False

            k_hop_neighbors_dict[i][current_layer] = current_hop_neighbors

        if reached_graph_diameter:
            break  # finished finding neighborhoods (to the depth that we want)
        else:
            current_layer += 1  # move out to next layer
    return k_hop_neighbors_dict, node_degrees


def get_degree_sequence(node_degrees, num_buckets, k_neighbors, max_degree):
    """
    Turn lists of neighbors into a degree sequence
    :param node_degrees: the degree of each node in a graph
    :param max_degree: max degree in all input graphs
    :param num_buckets: degree (node feature) binning
    :param k_neighbors: node's neighbors at a given layer
    :return: length-D list of ints (counts of nodes of each degree), where D is max degree in graph
    """
    if num_buckets is not None:
        degrees_vector = [0] * int(math.log(max_degree, num_buckets) + 1)
    else:
        degrees_vector = [0] * (max_degree + 1)

    # For each node in k-hop neighbors, count its degree
    for node in k_neighbors:
        weight = 1  # unweighted graphs supported here
        degree = node_degrees[node]
        # print("get_degree_sequence  %d %d", node, degree)
        if num_buckets is not None:
            try:
                degrees_vector[int(math.log(degree, num_buckets))] += weight
            except:
                print("Node %d has degree %d and will not contribute to feature distribution" % (node, degree))
        else:
            degrees_vector[degree] += weight
    return degrees_vector


def get_emb_sim_mat(node_emb1, node_emb2, stu_sim_measure="euclidean", att_sim_measure="euclidean", att_emb1=None,
                    att_emb2=None, gamma_struct=None, gamma_attr=None, num_top=None):
    """
    Convert distances (weighted by coefficients on structure and attributes) to similarities, the similarity score of
    each pair of nodes is computed based on embeddings with features of node attributes.
    """
    if node_emb2 is None:
        node_emb2 = node_emb1

    # KD tree with only top similarities computed, but without attributes consideration
    if num_top is not None and num_top != 0:
        kd_sim = kd_align(node_emb1, node_emb2, distance_metric=stu_sim_measure, num_top=num_top)
        return kd_sim

    # All pairwise distance computation
    if stu_sim_measure == "cosine":
        sim_mat = cosine_similarity(node_emb1, node_emb2)
    else:
        sim_mat = euclidean_distances(node_emb1, node_emb2)

    if att_emb1 is not None and att_emb2 is not None:
        if att_sim_measure == "cosine":
            att_sim_mat = cosine_similarity(att_emb1, att_emb2)
        elif att_sim_measure == "euclidean":
            att_sim_mat = euclidean_distances(att_emb1, att_emb2)
        else:
            att_sim_mat = pairwise_distances(att_emb1, att_emb2, sum_element_different)
        sim_mat = np.exp(-(gamma_struct * sim_mat + gamma_attr * att_sim_mat))
    else:
        sim_mat = np.exp(-sim_mat)
    return sim_mat


def sum_element_different(vector1, vector2):
    dis = np.sum(vector1 != vector2)
    return dis


def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=50):
    kd_tree = KDTree(emb2, metric=distance_metric)
    dist, ind = kd_tree.query(emb1, k=num_top)
    row = np.array([])

    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    # data = dist.flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()


def score(alignment_matrix, true_alignments=None):
    """The alignments are dictionary of the form node_in_graph 1 : node_in_graph2"""

    if true_alignments is None:  # assume it's just identity permutation
        return np.sum(np.diagonal(alignment_matrix))
    else:
        nodes_g1 = [int(node_g1) for node_g1 in true_alignments.keys()]
        nodes_g2 = [int(true_alignments[node_g1]) for node_g1 in true_alignments.keys()]
        return np.sum(alignment_matrix[nodes_g1, nodes_g2])


def get_ori_nodes_align_score(g1_nodes, g2_nodes, sim_matrix, true_alignments=None, top_k=None, score_weighted=False):
    """The alignment scores for all nodes in original networks"""

    nodes_num = sim_matrix.shape[0]
    if top_k is None:
        row_sums = sim_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 10e-6  # shouldn't affect much since dividing 0 by anything is 0
        sim_matrix = sim_matrix / row_sums[:, np.newaxis]  # normalize
        alignment_score = score(sim_matrix, true_alignments=true_alignments)
    else:
        alignment_score = 0
        if not sp.issparse(sim_matrix):
            sorted_indices = np.argsort(sim_matrix)

        for node_index in range(nodes_num):
            top_align_nodes_id = []
            node_id = g1_nodes[node_index]

            target_alignment = node_id  # default: the node should be aligned to itself
            if true_alignments is not None:  # if we have true alignments (which we require), use those for each node
                target_alignment = true_alignments[node_id]

            if sp.issparse(sim_matrix):
                row, possible_alignments, possible_values = sp.find(sim_matrix[node_index])
                node_sorted_indices = possible_alignments[possible_values.argsort()]
            else:
                node_sorted_indices = sorted_indices[node_index]

            # get all the nodes id (sub-nodes) at top k
            for n_index in node_sorted_indices[-top_k:]:
                node_id2 = g2_nodes[n_index]
                top_align_nodes_id.append(node_id2)

            if target_alignment in top_align_nodes_id:
                if score_weighted:
                    alignment_score += 1.0 / (
                            nodes_num - np.argwhere(sorted_indices[node_index] == target_alignment)[0])
                else:
                    alignment_score += 1
    return alignment_score


def get_sup_nodes_align_score(g1_super_nodes, g2_super_nodes, g1_super_nodes_dic, g2_super_nodes_dic,
                              g1_nodes_emb_dict, g2_nodes_emb_dict, g1_nodes_att_emb_dict, g2_nodes_att_emb_dict,
                              sup_sim_mat, true_alignments, top_k_sup=1, top_k_sub=1, args=None):
    """
    Calculate alignments score for super-nodes and their sub-nodes between two graphs with using possible node
    attributes features.
    """
    g1_sup_node_num = len(g1_super_nodes)
    total_sub_nodes_score = 0

    for n_ind1 in range(g1_sup_node_num):
        node_id1 = g1_super_nodes[n_ind1]
        sub_nodes1 = list(g1_super_nodes_dic[node_id1])  # get all the sub-nodes of sup-node 1
        sub_nodes_emb1 = [g1_nodes_emb_dict[k] for k in sub_nodes1]
        sub_nodes_att_emb1 = [g1_nodes_att_emb_dict[k] for k in sub_nodes1]

        # sort the nodes in g2 for each node of g1 according to similarity
        row, possible_alignments, similarities = sp.find(sup_sim_mat[n_ind1])
        node_sorted_indices = possible_alignments[similarities.argsort()]

        sub_nodes2 = []
        for n_ind in node_sorted_indices[-top_k_sup:]:
            node_id2 = g2_super_nodes[n_ind]
            # add all the sub-nodes in top k aligned super-nodes of g2
            sub_nodes2 = sub_nodes2 + list(g2_super_nodes_dic[node_id2])

        sub_nodes_emb2 = [g2_nodes_emb_dict[k] for k in sub_nodes2]
        sub_nodes_att_emb2 = [g2_nodes_att_emb_dict[k] for k in sub_nodes2]
        sub_nodes2_num = len(sub_nodes2)

        num_top = args.num_top
        if num_top > sub_nodes2_num:
            num_top = sub_nodes2_num
        sub_nodes_sim_mat = get_emb_sim_mat(np.asarray(sub_nodes_emb1),
                                            np.asarray(sub_nodes_emb2),
                                            att_emb1=np.asarray(sub_nodes_att_emb1),
                                            att_emb2=np.asarray(sub_nodes_att_emb2),
                                            gamma_struct=args.gamma_struc,
                                            gamma_attr=args.gamma_attr,
                                            # num_top=num_top
                                            )
        nodes_score = get_ori_nodes_align_score(sub_nodes1, sub_nodes2, sub_nodes_sim_mat, true_alignments, top_k_sub)
        total_sub_nodes_score += nodes_score

    return total_sub_nodes_score


def cal_sup_node_emb(g_nodes_emb_dict, sub_nodes_set):
    """Compute the embedding for super-nodes."""

    sub_nodes = list(sub_nodes_set)
    sub_nodes_len = len(sub_nodes)
    emb_sum = g_nodes_emb_dict[sub_nodes[0]]

    for j in range(1, sub_nodes_len):
        emb_sum = emb_sum + g_nodes_emb_dict[sub_nodes[j]]

    # emb_sum = np.divide(emb_sum, sub_nodes_len)
    emb_sum = 1 / (1 + np.exp(-emb_sum))
    return emb_sum


def get_nodes_dis(node_struct_emb1, node_struct_emb2):
    dis = np.linalg.norm(node_struct_emb1 - node_struct_emb2)
    return dis


def fast_match(g1_nodes, g2_nodes, g1_ori_n_emb, g2_ori_n_emb, g1_ori_n_num, g2_ori_n_num, nx_g1, nx_g2, node_deg_thr,
               sim_thr, compare_range):
    struct_zero_p = np.zeros(g1_ori_n_emb.shape[1])

    # construct two lists with distance
    g1_nodes_dis = [
        [g1_nodes[j], get_nodes_dis(g1_ori_n_emb[j], struct_zero_p), g1_ori_n_emb[j]] for j in
        range(g1_ori_n_num) if nx_g1.degree[g1_nodes[j]] > node_deg_thr]
    g2_nodes_dis = [
        [g2_nodes[j], get_nodes_dis(g2_ori_n_emb[j], struct_zero_p), g2_ori_n_emb[j]] for
        j in range(g2_ori_n_num) if nx_g2.degree[g2_nodes[j]] > node_deg_thr]

    g1_nodes_dis, g2_nodes_dis = filter_points_by_dis(g1_nodes_dis, g2_nodes_dis)
    g1_sorted_nodes, g2_sorted_nodes = filter_points_by_emb(g1_nodes_dis, g2_nodes_dis, compare_range, sim_thr)
    return g1_sorted_nodes, g2_sorted_nodes


def filter_points_by_dis(g1_nodes_dis_list, g2_nodes_dis_list):
    """ Delete the nodes that have same distance to zero point in two lists."""

    g1_nodes_dis_list = sorted(g1_nodes_dis_list, key=lambda tup: tup[1])
    g2_nodes_dis_list = sorted(g2_nodes_dis_list, key=lambda tup: tup[1])
    ind_start = 0
    ind_end = 1
    shift = 0
    nodes_num = len(g1_nodes_dis_list)
    dis1 = g1_nodes_dis_list[ind_start][1]

    # delete nodes in one network with save embeddings distance to a original node
    while ind_end < nodes_num:
        dis2 = g1_nodes_dis_list[ind_end][1]
        if dis1 == dis2:
            ind_end += 1
            continue
        elif (ind_end - ind_start) > 1:
            if (ind_start - shift) < 0:
                begin = 0
            else:
                begin = ind_start - shift
            if (ind_end + shift) > (nodes_num - 1):
                end = nodes_num - 1
            else:
                end = ind_end + shift
            # if the shift contain the node that has same distance with the next node
            dis3 = g1_nodes_dis_list[end - 1][1]
            dis4 = g1_nodes_dis_list[end][1]
            while dis3 == dis4:
                end += 1
                if end > nodes_num - 1:
                    end = nodes_num - 1
                    break
                dis4 = g1_nodes_dis_list[end][1]

            indices_del = [j for j in range(begin, end)]
            g1_nodes_dis_list = np.delete(g1_nodes_dis_list, indices_del, 0)
            g2_nodes_dis_list = np.delete(g2_nodes_dis_list, indices_del, 0)
            nodes_num = nodes_num - (end - begin)
            ind_start = begin
            ind_end = ind_start + 1
            dis1 = g1_nodes_dis_list[ind_start][1]  # reassign dis1 after deletion
        else:
            ind_start = ind_end
            ind_end += 1
            dis1 = g1_nodes_dis_list[ind_start][1]

    return g1_nodes_dis_list, g2_nodes_dis_list


def filter_points_by_emb(g1_nodes_dis_list, g2_nodes_dis_list, compare_range, sim_thr):
    """ Pick out every pair of nodes from two lists that have high embedding similarity."""

    if len(g1_nodes_dis_list) > len(g2_nodes_dis_list):
        fir_nodes_dis_list = g2_nodes_dis_list
        sec_nodes_dis_list = g1_nodes_dis_list
        reverse = True
    else:
        fir_nodes_dis_list = g1_nodes_dis_list
        sec_nodes_dis_list = g2_nodes_dis_list
        reverse = False
    sorted_nodes_fir = []
    sorted_nodes_sec = []
    n_num_compare = compare_range

    for ele1 in fir_nodes_dis_list:
        max_sim = -10
        best_match_node = None
        best_match_ind = None
        if len(sec_nodes_dis_list) < n_num_compare:
            n_num_compare = len(sec_nodes_dis_list)

        for j in range(n_num_compare):
            ele2 = sec_nodes_dis_list[j]
            sim = np.exp(-np.linalg.norm(ele1[2] - ele2[2]))
            if sim > max_sim:
                best_match_node = ele2[0]
                best_match_ind = j
                max_sim = sim

        if max_sim > sim_thr:
            sorted_nodes_fir.append(ele1[0])
            sorted_nodes_sec.append(best_match_node)
            sec_nodes_dis_list = np.delete(sec_nodes_dis_list, best_match_ind, 0)

    if reverse:
        return sorted_nodes_sec, sorted_nodes_fir
    else:
        return sorted_nodes_fir, sorted_nodes_sec


def get_node_attr_fea(g1_all_nodes_, g2_all_nodes_, att1, att2, att_num_):
    """ Construct features for node attributes."""

    g1_num = len(g1_all_nodes_)
    g2_num = len(g2_all_nodes_)
    g1_node_att_df = att1
    g2_node_att_df = att2

    if g2_num < g1_num:
        # Get names of indexes for which nodes that are saved by networkx when its degree becomes 0
        indexNames = g2_node_att_df[~g2_node_att_df['g2_n_id'].isin(g2_all_nodes_)].index
        # delete the nodes' attributes that are not in current graph
        g2_node_att_df.drop(indexNames, inplace=True)

    # Create the dictionary that defines the order for sorting
    sorterIndex = dict(zip(g1_all_nodes_, range(len(g1_all_nodes_))))
    # Generate a rank column that will be used to sort
    g1_node_att_df['n_id_rank'] = g1_node_att_df['g1_n_id'].map(sorterIndex)
    g1_node_att_df.sort_values(['n_id_rank'], ascending=[True], inplace=True)
    g1_node_att_df.drop('n_id_rank', 1, inplace=True)

    sorterIndex = dict(zip(g2_all_nodes_, range(len(g2_all_nodes_))))
    # Generate a rank column that will be used to sort
    g2_node_att_df['n_id_rank'] = g2_node_att_df['g2_n_id'].map(sorterIndex)
    g2_node_att_df.sort_values(['n_id_rank'], ascending=[True], inplace=True)
    g2_node_att_df.drop('n_id_rank', 1, inplace=True)

    g1_one_hot_df = None
    g2_one_hot_df = None

    for j in range(att_num_):
        att_name = 'att' + str(j + 1)

        if g1_one_hot_df is None:
            g1_one_hot_df = pd.get_dummies(g1_node_att_df[att_name], prefix=att_name)
        else:
            # use pd.concat to join the new columns with original df
            g1_one_hot_df = pd.concat([g1_one_hot_df, pd.get_dummies(g1_node_att_df[att_name], prefix=att_name)],
                                      axis=1)
        if g2_one_hot_df is None:
            g2_one_hot_df = pd.get_dummies(g2_node_att_df[att_name], prefix=att_name)
        else:
            # use pd.concat to join the new columns with original df
            g2_one_hot_df = pd.concat([g2_one_hot_df, pd.get_dummies(g2_node_att_df[att_name], prefix=att_name)],
                                      axis=1)

    g1_att = g1_one_hot_df.to_numpy()
    g2_att = g2_one_hot_df.to_numpy()
    return g1_att, g2_att
