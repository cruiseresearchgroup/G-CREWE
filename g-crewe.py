import networkx as nx
import numpy as np
import pandas as pd
import argparse
import copy
import pickle
import representation as rept
import regal.regal as reg
from merge import MERGE
from shrink import ShrinkMap
from settings import *
import gcn.embedding as gcn
import log


def parse_args(graph_name_, edge_noise_, att_noise_):
    if graph_name == "Brightkite":
        path = 'data/Brightkite/'
        file_g1 = 'Brightkite_friendship1.txt'
        file_g2 = 'Brightkite_friendship1_' + edge_noise_ + '.txt'
    else:
        print("no graphs")
        return

    file_matching = path + 'perm_nodes_matching.pkl'
    file_attr1 = path + 'g1_node_attrs.csv'
    file_attr2 = path + 'g2_node_attrs_' + att_noise_ + '.csv'

    parser = argparse.ArgumentParser(description="Configuration")
    parser.add_argument('--graph_name', nargs='?', default=graph_name_, help='')
    parser.add_argument('--graph1_path', nargs='?', default=path + file_g1, help='')
    parser.add_argument('--graph2_path', nargs='?', default=path + file_g2, help='')
    parser.add_argument('--g1_attr_path', nargs='?', default=file_attr1, help='')
    parser.add_argument('--g2_attr_path', nargs='?', default=file_attr2, help='')
    parser.add_argument('--true_align_path', nargs='?', default=file_matching, help='')
    parser.add_argument('--perm_edge_noise', nargs='?', default=edge_noise_, help='')
    parser.add_argument('--attributes', nargs='?', default=True, help='Whether node attributes are available.')
    parser.add_argument('--attr_values', type=int, default=3,
                        help='Number of attribute values. Only used if synthetic attributes are generated')
    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
    parser.add_argument('--k', type=int, default=10, help='Control size of landmarks to sample. Default is 10.')
    parser.add_argument('--max_layer', type=int, default=2, help='Calculation until the layer for xNetMF.')
    parser.add_argument('--num_buckets', default=2, type=float, help="base of log for degree (node feature) binning")
    parser.add_argument('--alpha', type=float, default=0.01, help="Discount factor for neighbors at further hops")
    parser.add_argument('--gamma_struc', type=float, default=1, help="Weight on structural similarity")
    parser.add_argument('--gamma_attr', type=float, default=1, help="Weight on attribute similarity")
    parser.add_argument('--num_top', type=int, default=50,
                        help="top similarities computed with KD-tree. 0 - compute all pairwise similarities.")
    parser.add_argument('--varphi', type=float, default=0.2, help='compression ratio for input graph')
    parser.add_argument('--merge_mode', type=int, default=3, help='1. Simple merge; 2. Han; 3. MERGE; 4. Shrink')
    parser.add_argument('--a_node_select_mode', type=int, default=0,
                        help='How to select the first node for compression? 0. no; 1. embeddings.')
    parser.add_argument('--b_node_select_mode', type=int, default=1,
                        help='How to select the second node for compression? 0. no; 1. small neighbours.')
    parser.add_argument('--embed_mode', type=int, default=1, help="1. gcn; 2. xNetMF")
    parser.add_argument('--top_a_align', type=int, default=1,
                        help="Default number of top alignments considered between original nodes.")
    parser.add_argument('--top_a_align_sup', type=int, default=50,
                        help="Default number of top alignments considered between super-nodes.")
    parser.add_argument('--eta', type=int, default=15,
                        help="Node degree threshold for fast alignment in making guiding lists.")
    parser.add_argument('--omega', type=float, default=0.99,
                        help="Nodes similarity threshold for fast alignment in making guiding lists.")
    parser.add_argument('--lambda_', type=int, default=100,
                        help="Number of top nodes in second list for pairing in making guiding lists.")

    return parser.parse_args()


if __name__ == "__main__":
    # initialize setting for gcn model
    net_settings = gcn.ini_net_settings(learning_rate=0.01, hidden1=16, dropout=0.5, layer_num=2)

    graph_name = "Brightkite"
    edge_noise_str = '0.01'
    attr_noise_str = '0.1'
    args = parse_args(graph_name, edge_noise_str, attr_noise_str)

    attr_num = args.attr_values
    embed_way = args.embed_mode
    comp_way = args.merge_mode
    f_match_deg_thr = args.eta
    f_match_sim_thr = args.omega
    f_match_compare_range = args.lambda_
    top_a_align = args.top_a_align
    top_a_align_sup_def = args.top_a_align_sup

    # read true alignments
    f = open(args.true_align_path, 'rb')
    true_alignments = pickle.load(f)
    f.close()

    nx_ori_g1 = nx.read_edgelist(args.graph1_path, nodetype=int, comments="%")
    nx_ori_g2 = nx.read_edgelist(args.graph2_path, nodetype=int, comments="%")
    g1_all_nodes = list(nx_ori_g1.nodes())
    g2_all_nodes = list(nx_ori_g2.nodes())
    g1_all_ori_nodes_num = nx_ori_g1.number_of_nodes()
    g2_all_ori_nodes_num = nx_ori_g2.number_of_nodes()
    max_node1 = np.max(g1_all_nodes)
    max_node2 = np.max(g2_all_nodes)
    ini_super_node_id = max(max_node1, max_node2) + 1
    all_ori_nodes = g1_all_nodes + g2_all_nodes
    g1_nodes_att_emb, g2_nodes_att_emb = None, None
    g1_nodes_att_emb_dict, g2_nodes_att_emb_dict = None, None

    if args.attributes:
        g1_attr_df = pd.read_csv(args.g1_attr_path)
        g2_attr_df = pd.read_csv(args.g2_attr_path)
        g1_nodes_att_emb, g2_nodes_att_emb = rept.get_node_attr_fea(g1_all_nodes, g2_all_nodes,
                                                                    g1_attr_df, g2_attr_df,
                                                                    attr_num)
        g1_nodes_att_emb_dict = dict(zip(g1_all_nodes, g1_nodes_att_emb))
        g2_nodes_att_emb_dict = dict(zip(g2_all_nodes, g2_nodes_att_emb))

    # -----------------learning embeddings for original nodes-----------------
    rept_settings = rept.ini_settings(args)
    emb_start_time = log.get_time()
    if embed_way == 1:
        # get nodes structural embedding for original graphs by gcn
        all_ori_nodes_emb = rept.get_embed_gcn(nx_ori_g1, nx_ori_g2,
                                               g1_all_ori_nodes_num,
                                               g2_all_ori_nodes_num,
                                               rept_settings,
                                               net_settings)
    else:
        ori_rep_g1 = Graph(nx.adjacency_matrix(nx_ori_g1), all_nodes_num=g1_all_ori_nodes_num)
        ori_rep_g2 = Graph(nx.adjacency_matrix(nx_ori_g2), all_nodes_num=g2_all_ori_nodes_num)
        # get nodes structural embedding for original graphs by xNetMF
        all_ori_nodes_emb = reg.get_embed_reg(ori_rep_g1, ori_rep_g2, None, None, rept_settings)
    emb_end_time = log.get_time()

    # -----------------generate guiding lists based on node embeddings-----------------
    sort_start_time = log.get_time()
    g1_all_ori_nodes_emb = all_ori_nodes_emb[:g1_all_ori_nodes_num]
    g2_all_ori_nodes_emb = all_ori_nodes_emb[g1_all_ori_nodes_num:]
    g1_ori_nodes_emb_dict = dict(zip(g1_all_nodes, g1_all_ori_nodes_emb))
    g2_ori_nodes_emb_dict = dict(zip(g2_all_nodes, g2_all_ori_nodes_emb))
    g1_gui_list = []
    g2_gui_list = []

    while len(g1_gui_list) < 1:  # if no nodes are matched in process of making guiding lists
        g1_gui_list, g2_gui_list = rept.fast_match(g1_all_nodes, g2_all_nodes,
                                                   g1_all_ori_nodes_emb, g2_all_ori_nodes_emb,
                                                   g1_all_ori_nodes_num, g2_all_ori_nodes_num,
                                                   nx_ori_g1, nx_ori_g2,
                                                   f_match_deg_thr, f_match_sim_thr, f_match_compare_range)
        f_match_sim_thr -= 0.01  # decrease the similarity threshold for find a matched node in another list
        f_match_compare_range += 20  # increase the range for find a matched node in another list

    # check matching accurate for nodes among guiding lists
    correct_num = 0
    count = 0
    for fir_node_id in g1_gui_list:
        target_node_id = true_alignments[fir_node_id]
        align_node_id = g2_gui_list[count]
        if target_node_id == align_node_id:
            correct_num += 1
        count += 1
    print("total num of nodes in guiding list:", str(len(g1_gui_list)))
    print("sorted nodes alignment accuracy:", str(correct_num / len(g1_gui_list)))

    g1_gui_list.reverse()
    g2_gui_list.reverse()
    sort_end_time = log.get_time()

    comp_start_time = log.get_time()
    if comp_way == 3:
        comp_net1 = MERGE(copy.deepcopy(nx_ori_g1), args, ini_super_node_id, g1_all_ori_nodes_num, g1_gui_list)
        comp_net1.execute()
        print("Compress first graph end")
        comp_net2 = MERGE(copy.deepcopy(nx_ori_g2), args, comp_net1.ini_super_node_id, g2_all_ori_nodes_num,
                          g2_gui_list)
        comp_net2.execute()
        print("Compress second graph end")
    elif comp_way == 4:
        comp_net1 = ShrinkMap(copy.deepcopy(nx_ori_g1), args, ini_super_node_id, g1_all_ori_nodes_num, g1_gui_list)
        comp_net1.execute()
        print("Compress first graph end")
        comp_net2 = ShrinkMap(copy.deepcopy(nx_ori_g2), args, comp_net1.ini_super_node_id, g2_all_ori_nodes_num,
                              g2_gui_list)
        comp_net2.execute()
        print("Compress second graph end")
    else:
        comp_net1 = None
        comp_net2 = None
    comp_end_time = log.get_time()

    g1_comp_all_nodes = list(comp_net1.graph.nodes())
    g2_comp_all_nodes = list(comp_net2.graph.nodes())
    g1_super_nodes_dic = comp_net1.super_nodes_dict
    g2_super_nodes_dic = comp_net2.super_nodes_dict
    g1_super_nodes = list(g1_super_nodes_dic.keys())
    g2_super_nodes = list(g2_super_nodes_dic.keys())
    g1_uncomp_ori_nodes = list(set(g1_comp_all_nodes) - set(g1_super_nodes))
    g2_uncomp_ori_nodes = list(set(g2_comp_all_nodes) - set(g2_super_nodes))

    # get embedding for original uncompressed nodes and super nodes
    emb2_start_time = log.get_time()
    g1_sup_nodes_emb = [rept.cal_sup_node_emb(g1_ori_nodes_emb_dict, g1_super_nodes_dic[n_id])
                        for n_id in g1_super_nodes]
    g2_sup_nodes_emb = [rept.cal_sup_node_emb(g2_ori_nodes_emb_dict, g2_super_nodes_dic[n_id])
                        for n_id in g2_super_nodes]

    g1_uncomp_nodes_emb = [g1_ori_nodes_emb_dict[n_id] for n_id in g1_uncomp_ori_nodes]
    g2_uncomp_nodes_emb = [g2_ori_nodes_emb_dict[n_id] for n_id in g2_uncomp_ori_nodes]
    emb2_end_time = log.get_time()

    if args.attributes:
        g1_uncomp_nodes_att_emb = [g1_nodes_att_emb_dict[n_id] for n_id in g1_uncomp_ori_nodes]
        g2_uncomp_nodes_att_emb = [g2_nodes_att_emb_dict[n_id] for n_id in g2_uncomp_ori_nodes]
    else:
        g1_uncomp_nodes_att_emb = None
        g2_uncomp_nodes_att_emb = None

    sim_mat_start_time = log.get_time()
    uncomp_nodes_sim_mat = rept.get_emb_sim_mat(np.asarray(g1_uncomp_nodes_emb),
                                                np.asarray(g2_uncomp_nodes_emb),
                                                att_emb1=np.asarray(g1_uncomp_nodes_att_emb),
                                                att_emb2=np.asarray(g2_uncomp_nodes_att_emb),
                                                gamma_struct=args.gamma_struc,
                                                gamma_attr=args.gamma_attr,
                                                # num_top=args.num_top
                                                )
    sup_nodes_sim_mat = rept.get_emb_sim_mat(np.asarray(g1_sup_nodes_emb),
                                             np.asarray(g2_sup_nodes_emb),
                                             num_top=args.num_top
                                             )
    sim_mat_end_time = log.get_time()

    print("---------Running Time--------")
    log.time_diff_tag("Time - making guiding lists", sort_start_time, sort_end_time)
    log.time_diff_tag("Time - compressing nodes", comp_start_time, comp_end_time)
    log.time_diff_tag("Time - original nodes embedding", emb_start_time, emb_end_time)
    log.time_diff_tag("Time - sup-nodes embedding", emb2_start_time, emb2_end_time)
    log.time_diff_tag("Time - similarity calculation", sim_mat_start_time, sim_mat_end_time)

    # ---------------------------get alignment accuracy---------------------------
    sup_nodes_num = len(g2_super_nodes)
    noise_grading = 0.03
    # increase the number of top similar super-nodes in second graph for comparision according to noise levels
    if float(edge_noise_str) >= noise_grading:
        top_k_sup = int(sup_nodes_num * 0.3)
    elif 0.01 < float(edge_noise_str) < noise_grading:
        top_k_sup = int(sup_nodes_num * 0.4)
    else:
        top_k_sup = int(sup_nodes_num * 0.5)

    if top_k_sup < top_a_align_sup_def:
        top_k_sup = top_a_align_sup_def

    startTime = log.get_time()
    uncomp_nodes_score = rept.get_ori_nodes_align_score(g1_uncomp_ori_nodes, g2_uncomp_ori_nodes,
                                                        uncomp_nodes_sim_mat, true_alignments,
                                                        top_a_align)
    sup_nodes_score = rept.get_sup_nodes_align_score(g1_super_nodes, g2_super_nodes,
                                                     g1_super_nodes_dic, g2_super_nodes_dic,
                                                     g1_ori_nodes_emb_dict, g2_ori_nodes_emb_dict,
                                                     g1_nodes_att_emb_dict, g2_nodes_att_emb_dict,
                                                     sup_nodes_sim_mat, true_alignments,
                                                     top_k_sup=top_k_sup, top_k_sub=top_a_align,
                                                     args=args)
    log.time_diff_tag("Time - node alignments", startTime, log.get_time())
    print("---------Top %d align score---------" % top_a_align)
    final_align_score = round((uncomp_nodes_score + sup_nodes_score) / g1_all_ori_nodes_num, 4)
    log.info_tag("Final alignment score", final_align_score)
