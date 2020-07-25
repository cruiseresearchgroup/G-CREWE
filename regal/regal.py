"""
Copyright (c) 2018, Mark Heimann.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import networkx as nx
import sklearn.metrics.pairwise
from scipy.sparse import coo_matrix
from sklearn.neighbors import KDTree
from settings import *
import gcn.embedding as gcn
import scipy.sparse as sp
import numpy as np
import math


def align_graphs(args, nx_g1, nx_g2, g1_node_num, g2_node_num, g1_node_attrs, g2_node_attrs, embed_way="reg"):
    """
    :param g2_node_attrs:
    :param g1_node_attrs:
    :param args:
    :param nx_g1:
    :param nx_g2:
    :param g1_node_num:
    :param g2_node_num:
    :param embed_way: "xNetMF" or "gcn"
    :return: nodes similarities matrix
    """
    # Learn embeddings and save to output
    print("learning representations...")
    num_buckets = args.num_buckets  # base of log for log scale
    if num_buckets == 1:
        num_buckets = None
    rep_settings = ParaSettings(max_layer=args.max_layer, alpha=args.alpha, k=args.k, num_buckets=num_buckets,
                                normalize=True, gamma_struc=args.gamma_struc, gamma_attr=args.gamma_attr)

    adj1 = nx.adjacency_matrix(nx_g1)
    adj2 = nx.adjacency_matrix(nx_g2)
    print("get adj matrix for two graphs")
    rep_g1 = Graph(adj1, all_nodes_num=g1_node_num)
    rep_g2 = Graph(adj2, all_nodes_num=g2_node_num)

    if embed_way == "gcn":
        combine_net = nx.compose(nx_g1, nx_g2)
        combine_adj = nx.adjacency_matrix(combine_net)
        similar_matrix = get_simmatrix_gcn(args.num_top, rep_g1, rep_g2, rep_settings, combine_adj)
    else:
        similar_matrix = get_simmatrix(args.num_top, rep_g1, rep_g2, g1_node_attrs, g2_node_attrs, rep_settings)
    return similar_matrix


def get_embed_reg(rep_g1, rep_g2, g1_node_attrs, g2_node_attrs, rep_settings):
    feature_matrix, num_features = get_nodes_feature(rep_g1, rep_g2, rep_settings)
    total_nodes_num = rep_g1.all_nodes_num + rep_g2.all_nodes_num
    embeddings = cal_reps_with_landmark(feature_matrix, rep_g1.all_nodes_num, rep_settings, total_nodes_num,
                                        g1_node_attrs, g2_node_attrs)
    return embeddings


def get_simmatrix(num_top, rep_g1, rep_g2, g1_node_attrs, g2_node_attrs, rep_settings):
    embeddings = get_embed_reg(rep_g1, rep_g2, g1_node_attrs, g2_node_attrs, rep_settings)
    emb1 = embeddings[:rep_g1.all_nodes_num]  # embeddings for nodes in graph 1
    emb2 = embeddings[rep_g1.all_nodes_num:]  # embeddings for nodes in graph 2
    similar_matrix = get_embed_similarity(emb1, emb2, num_top=None)
    return similar_matrix


def get_simmatrix_gcn(num_top, rep_g1, rep_g2, rep_settings, combine_adj):
    feature_matrix, num_features = get_nodes_feature(rep_g1, rep_g2, rep_settings)

    # Initialize the settings for neural networks and layers to get embeddings for a combine network
    settings = gcn.ini_net_settings(learning_rate=0.01, hidden1=16, dropout=0.5)
    embeddings = gcn.get_embeddings(combine_adj, feature_matrix,
                                    [rep_g1.all_nodes_num + rep_g2.all_nodes_num, num_features * 2], settings)
    emb1 = embeddings[:rep_g1.all_nodes_num]  # embeddings for nodes in graph 1
    emb2 = embeddings[rep_g1.all_nodes_num:]  # embeddings for nodes in graph 2
    similar_matrix = get_embed_similarity(emb1, emb2, num_top=num_top)
    return similar_matrix


def cal_reps_with_landmark(feature_matrix, g1_nodes_num, rep_settings, total_nodes_num, g1_node_attrs, g2_node_attrs):
    # Get landmark nodes(to compute all pairwise similarities to in Nystrom approx)
    if rep_settings.p is None:
        p = int(rep_settings.k * math.log(g1_nodes_num, 2))  # k*log(n), where k = 10
        rep_settings.p = min(p, g1_nodes_num)  # don't return larger dimensionality than # of nodes
        print("feature dimensionality is " + str(rep_settings.p))
    elif rep_settings.p > g1_nodes_num:
        print("Warning: dimensionality greater than number of nodes. Reducing to n")
        rep_settings.p = g1_nodes_num

    if g1_node_attrs is not None and g2_node_attrs is not None:
        all_node_attrs = np.concatenate((g1_node_attrs, g2_node_attrs), axis=0)
    else:
        all_node_attrs = None

    # permute uniformly at random
    landmarks = np.random.permutation(range(total_nodes_num))[:rep_settings.p]
    # Explicitly compute similarities of all nodes to these landmarks. Discontinuous node ids in two graphs are
    # matched to 0 to max node num.
    C = np.zeros((total_nodes_num, rep_settings.p))

    for node_index in range(total_nodes_num):  # for each of nodes in all graphs
        for landmark_index in range(rep_settings.p):  # for each of p landmarks
            if all_node_attrs is not None:
                attr1 = all_node_attrs[node_index]
                attr2 = all_node_attrs[landmarks[landmark_index]]
            else:
                attr1 = None
                attr2 = None

            # select the p-th landmark
            C[node_index, landmark_index] = get_features_similarity(rep_settings, feature_matrix[node_index],
                                                                    feature_matrix[landmarks[landmark_index]],
                                                                    attr1, attr2)
    # Compute Nystrom-based node embeddings
    W_pin_v = np.linalg.pinv(C[landmarks])
    U, X, V = np.linalg.svd(W_pin_v)
    W_fac = np.dot(U, np.diag(np.sqrt(X)))
    represent = np.dot(C, W_fac)

    # Post-processing step to normalize embeddings (true by default, for use with REGAL)
    if rep_settings.normalize:
        represent = represent / np.linalg.norm(represent, axis=1).reshape((represent.shape[0], 1))
    return represent


def get_nodes_feature(rep_g1, rep_g2, rep_settings):
    """
    Get structural features for nodes in a graph based on degree sequences of neighbors
    :return: Discontinuous node ids in graph are matched to 0 to max node num in feature_matrix
    """
    # Get k-hop neighbors of all nodes. And degree of each node in graph, index starts from 0 to maximum.
    rep_g1_k_hop_neighbors, rep_g1_node_degrees = get_k_hop_neighbors(rep_g1, rep_settings)
    rep_g2_k_hop_neighbors, rep_g2_node_degrees = get_k_hop_neighbors(rep_g2, rep_settings)
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
    return feature_matrix, num_features


def get_k_hop_neighbors(rep_g, rep_settings):
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
    for node in range(rep_g.all_nodes_num):
        neighbors = np.nonzero(rep_g.G_adj[node])[-1].tolist()  # column indices of non-zero elements
        node_degrees.append(len(neighbors))

        if len(neighbors) == 0:
            print("Warning: node %d is disconnected " + str(node))
            k_hop_neighbors_dict[node] = {0: {node}, 1: set()}
        else:
            if type(neighbors[0]) is list:
                neighbors = neighbors[0]
            k_hop_neighbors_dict[node] = {0: {node}, 1: set(neighbors) - {node}}
        # For each node, keep track of neighbors we've already seen
        all_neighbors_traversed[node] = {node}.union(k_hop_neighbors_dict[node][1])

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


def get_features_similarity(rep_settings, vec1, vec2, attr1, attr2):
    """
    convert distances (weighted by coefficients on structure and attributes) to similarities
    :return:  number between 0 and 1 representing their similarity
    """
    # compare distances between structural identities
    dist = rep_settings.gamma_struc * np.linalg.norm(vec1 - vec2)

    if attr1 is not None and attr2 is not None:
        attr_dist = 0
        for i in range(attr1.shape[0]):
            # distance is number of disagreeing attributes
            if attr1[i] != attr2[i]:
                attr_dist += 1
        dist += rep_settings.gamma_attr * attr_dist
    return np.exp(-dist)


def get_embed_similarity(embed1, embed2, sim_measure="euclidean", num_top=None):
    """
    Score alignments based on embeddings of two graphs
    """
    if embed2 is None:
        embed2 = embed1

    if num_top is not None and num_top != 0:  # KD tree with only top similarities computed
        kd_sim = kd_align(embed1, embed2, distance_metric=sim_measure, num_top=num_top)
        return kd_sim

    # All pairwise distance computation
    if sim_measure == "cosine":
        similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(embed1, embed2)
    else:
        similarity_matrix = sklearn.metrics.pairwise.euclidean_distances(embed1, embed2)
        similarity_matrix = np.exp(-similarity_matrix)
    return similarity_matrix


def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=50):
    kd_tree = KDTree(emb2, metric=distance_metric)
    dist, ind = kd_tree.query(emb1, k=num_top)
    row = np.array([])

    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()


def get_alignment_score(g1_nodes, g2_nodes, align_matrix, true_alignments=None, top_k=None, top_k_score_weighted=False):
    n_nodes = align_matrix.shape[0]
    correct_nodes = []

    if top_k is None:
        row_sums = align_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 10e-6  # shouldn't affect much since dividing 0 by anything is 0
        align_matrix = align_matrix / row_sums[:, np.newaxis]  # normalize
        alignment_score = score(align_matrix, true_alignments=true_alignments)
    else:
        alignment_score = 0
        if not sp.issparse(align_matrix):
            sorted_indices = np.argsort(align_matrix)

        for node_index in range(n_nodes):
            top_align_nodes_id2 = []
            node_id = g1_nodes[node_index]

            target_alignment = node_id  # default: the node should be aligned to itself
            if true_alignments is not None:  # if we have true alignments (which we require), use those for each node
                target_alignment = true_alignments[node_id]

            if sp.issparse(align_matrix):
                row, possible_alignments, possible_values = sp.find(align_matrix[node_index])
                node_sorted_indices = possible_alignments[possible_values.argsort()]
            else:
                node_sorted_indices = sorted_indices[node_index]

            # get all the nodes id (sub-nodes) at top k
            for n_index in node_sorted_indices[-top_k:]:
                node_id2 = g2_nodes[n_index]
                top_align_nodes_id2.append(node_id2)

            if target_alignment in top_align_nodes_id2:
                if top_k_score_weighted:
                    alignment_score += 1.0 / (n_nodes - np.argwhere(sorted_indices[node_index] == target_alignment)[0])
                else:
                    alignment_score += 1
                # correct_nodes.append(node_index)
        alignment_score /= float(n_nodes)
    return alignment_score, set(correct_nodes)


def score(alignment_matrix, true_alignments=None):
    """
    alignments are dictionary of the form node_in_graph 1 : node_in_graph2
    """
    if true_alignments is None:  # assume it's just identity permutation
        return np.sum(np.diagonal(alignment_matrix))
    else:
        nodes_g1 = [int(node_g1) for node_g1 in true_alignments.keys()]
        nodes_g2 = [int(true_alignments[node_g1]) for node_g1 in true_alignments.keys()]
        return np.sum(alignment_matrix[nodes_g1, nodes_g2])
