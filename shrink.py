"""
Amin Sadri, Flora D. Salim and Yongli Ren
Copyright (c) 2017, RMIT University.

The original source code provided in the paper:
Sadri, A., Salim, F. D., Ren, Y., Zameni, M., Chan, J., & Sellis, T. (2017). Shrink: Distance preserving graph
compression. Information Systems, 69(June), 180–193. https://doi.org/10.1016/j.is.2017.06.001
"""

import sys


class ShrinkMap:
    """
    Compress a graph for preserving shortest distance (original)
    """

    def __init__(self, graph, args, ini_super_node_id=0, node_num=0, nodes_list_comp=None):
        self.graph = graph  # a networkx graph
        self.guide_list = nodes_list_comp  # all node ids of the graph or node ids in each cluster
        self.super_nodes_dict = {}  # a dictionary to save super-nodes and their sub-nodes
        self.ini_super_node_id = ini_super_node_id
        self.ori_node_num = node_num
        self.min_neigh_threshold = 1
        self.merge_mode = args.merge_mode
        self.a_node_select_mode = args.a_node_select_mode
        self.b_node_select_mode = args.b_node_select_mode
        self.compress_ratio = args.compress_ratio

    def execute(self):
        comp_size = int(self.compress_ratio * self.ori_node_num)
        merge_node_id = -1
        if self.a_node_select_mode == 0:
            while comp_size > 0:
                merge_node_id += 1
                try:
                    a_id = self.guide_list[merge_node_id]
                except IndexError:
                    merge_node_id = -1
                    continue

                if self.merge(a_id):
                    comp_size -= 1

    def merge(self, aId):
        if not self.graph.has_node(aId):  # if not - it is zero, empty or False
            return False

        a_neigh_weights = []  # weight of all neighbours of node a
        b_neigh_weights = []  # weight of all neighbours of node b
        a_only_neighs = []  # non-shared neighbours of node a
        b_only_neighs = []  # non-shared neighbours of node b
        share_neighs = []  # shared neighbours of node a and b
        a_only_neigh_weights = []  # weight of neighbours of node a only
        b_only_neigh_weights = []  # weight of neighbours of node b only
        a_share_neigh_weights = []  # weight of shared neighbours from a
        b_share_neigh_weights = []  # weight of shared neighbours from b
        min_weight = sys.maxsize
        bId = 0

        a_neighbours = list(self.graph.neighbors(aId))  # id of all neighbours of node a
        a_neigh_num = len(a_neighbours)
        if not a_neigh_num:
            return False

        # get weights of node a neighbours
        for i in range(a_neigh_num):
            neighId = a_neighbours[i]
            a_neigh_weights.append(self.get_weight(aId, neighId))
            if a_neigh_weights[i] < min_weight:
                min_weight = a_neigh_weights[i]
                bId = neighId

        b_neighbours = list(self.graph.neighbors(bId))  # id of all neighbours of node b
        b_neigh_num = len(b_neighbours)

        # -------------------------------------------------------------------------------
        #  pass selection metrics for node a and b
        # -------------------------------------------------------------------------------
        if self.b_node_select_mode == 1:
            if self.min_neigh_threshold < a_neigh_num + b_neigh_num:
                self.update_threshold(1)
                return False
            else:
                self.update_threshold(-1)

        # -------------------------------------------------------------------------------
        #  check whether node a and b is super-node and limit their num of sub-nodes
        # -------------------------------------------------------------------------------
        if aId in self.super_nodes_dict and bId in self.super_nodes_dict:  # if A and B are super-node
            self.super_nodes_dict[bId] = self.super_nodes_dict[bId] + self.super_nodes_dict[aId]
            self.super_nodes_dict.pop(aId)  # delete super node from storage list
            cId = bId
        elif aId in self.super_nodes_dict:  # if only A is super-node
            self.super_nodes_dict[aId].append(bId)
            cId = aId
        elif bId in self.super_nodes_dict:  # if only B is super-node
            self.super_nodes_dict[bId].append(aId)
            cId = bId
        else:  # create a new super-node when A and B are all not super nodes
            self.ini_super_node_id += 1
            cId = self.ini_super_node_id
            self.super_nodes_dict[cId] = [aId, bId]

        # -------------------------------------------------------------------------------
        #  separate three sets (right, share, left) of neighbours for node a and b
        # -------------------------------------------------------------------------------
        for i in range(b_neigh_num):
            neighId = b_neighbours[i]
            b_neigh_weights.append(self.get_weight(bId, neighId))  # get weights of node b neighbours

        p = 0
        r = 0
        q = 0
        for i in range(a_neigh_num):
            aNeighId = a_neighbours[i]
            if aNeighId == bId:
                continue
            if aNeighId not in b_neighbours:
                a_only_neighs.append(aNeighId)
                a_only_neigh_weights.append(a_neigh_weights[i])
                p += 1
            else:
                share_neighs.append(aNeighId)
                a_share_neigh_weights.append(a_neigh_weights[i])
                b_share_neigh_weights.append(self.get_weight(bId, aNeighId))
                r += 1
        for j in range(b_neigh_num):
            bNeighId = b_neighbours[j]
            if bNeighId == aId:
                continue
            if bNeighId not in a_neighbours:
                b_only_neighs.append(bNeighId)
                b_only_neigh_weights.append(b_neigh_weights[j])
                q += 1

        # remove old nodes with their edges, and add super-node to graph
        self.graph.remove_node(aId)
        self.graph.remove_node(bId)
        self.graph.add_node(cId)
        self.guide_list.remove(aId)
        if bId in self.guide_list:
            self.guide_list.remove(bId)
        self.guide_list.append(cId)

        # # get and add edges with weights for the new super-node
        # c_neigh_weights = self.calculateWeights(a_only_neigh_weights, a_share_neigh_weights, b_share_neigh_weights,
        #                                         b_only_neigh_weights, p, r, q, min_weight, a_sub_nodes_num,
        #                                         b_sub_nodes_num)
        # index = 0
        # for nodeId in a_only_neighs:
        #     self.graph.add_edge(cId, nodeId, weight=c_neigh_weights[index])
        #     index += 1
        # for nodeId in share_neighs:
        #     self.graph.add_edge(cId, nodeId, weight=c_neigh_weights[index])
        #     index += 1
        # for nodeId in b_only_neighs:
        #     self.graph.add_edge(cId, nodeId, weight=c_neigh_weights[index])
        #     index += 1

        # Use default value for the new weights of super-nodes' connections
        index = 0
        default_weight = 1
        for nodeId in a_only_neighs:
            self.graph.add_edge(cId, nodeId, weight=default_weight)
            index += 1
        for nodeId in share_neighs:
            self.graph.add_edge(cId, nodeId, weight=default_weight)
            index += 1
        for nodeId in b_only_neighs:
            self.graph.add_edge(cId, nodeId, weight=default_weight)
            index += 1

        return True

    def update_threshold(self, diff):
        self.min_neigh_threshold += diff

    def get_weight(self, aId, bId):
        weight = self.graph[aId][bId]['weight']
        return weight

    def cal_weights(self, aOnlyNeighWeights, aShareNeighWeights, bShareNeighWeights, bOnlyNeighWeights, p, r, q,
                    minNeighWeight, aSubNodesNum, bSubNodesNum):
        cNeighWeights = []

        if self.merge_mode == 1:  # Simple merge
            for i in range(p):
                cNeighWeights.append(aOnlyNeighWeights[i])
            for i in range(r):
                cNeighWeights.append(bShareNeighWeights[i])
            for i in range(q):
                cNeighWeights.append(bOnlyNeighWeights[i])
        elif self.merge_mode == 2:  # Han
            for i in range(p):
                cNeighWeights.append(aOnlyNeighWeights[i])
            for i in range(r):
                cNeighWeights.append((aSubNodesNum * aShareNeighWeights[i] + bSubNodesNum * bShareNeighWeights[i]) / (
                        aSubNodesNum + bSubNodesNum))
            for i in range(q):
                cNeighWeights.append(bOnlyNeighWeights[i])
        elif self.merge_mode == 4:  # shrink
            index = 0
            edge_num = p + r + q
            C = []
            aShareNeighWeitSum = 0
            bShareNeighWeitSum = 0

            aOnlyNeighWeitSum = sum(aOnlyNeighWeights)
            bOnlyNeighWeitSum = sum(bOnlyNeighWeights)
            for i in range(r):
                aShareNeighWeitSum += min(aShareNeighWeights[i], minNeighWeight + bShareNeighWeights[i])
                bShareNeighWeitSum += min(bShareNeighWeights[i], minNeighWeight + aShareNeighWeights[i])

            for i in range(p):
                C.append((edge_num - 2) * aOnlyNeighWeights[i] + aOnlyNeighWeitSum + bOnlyNeighWeitSum)
                C[index] += q * minNeighWeight
                C[index] += aShareNeighWeitSum
                index += 1
            for i in range(r):
                C.append(aOnlyNeighWeitSum + bOnlyNeighWeitSum)
                C[index] += p * min(aShareNeighWeights[i], minNeighWeight + bShareNeighWeights[i])
                C[index] += q * min(bShareNeighWeights[i], minNeighWeight + aShareNeighWeights[i])
                for j in range(r):
                    if i == j:
                        continue
                    minEdge1 = min(aShareNeighWeights[j] + bShareNeighWeights[i] + minNeighWeight, aShareNeighWeights[
                        i] + bShareNeighWeights[j] + minNeighWeight)
                    minEdge2 = min(aShareNeighWeights[j] + aShareNeighWeights[i],
                                   bShareNeighWeights[i] + bShareNeighWeights[j])
                    C[index] += min(minEdge1, minEdge2)
                index += 1
            for i in range(q):
                C.append((edge_num - 2) * bOnlyNeighWeights[i] + aOnlyNeighWeitSum + bOnlyNeighWeitSum)
                C[index] += p * minNeighWeight
                C[index] += bShareNeighWeitSum
                index += 1

            if edge_num > 2:
                S = sum(C)
                S = S / (2 * edge_num - 2)
                for i in range(edge_num):
                    cNeighWeights.append((C[i] - S) / (edge_num - 2))
            elif edge_num == 2:
                cNeighWeights.append(C[0] / 2)
                cNeighWeights.append(C[1] / 2)
            elif edge_num == 1:
                if p != 0:
                    cNeighWeights.append(aOnlyNeighWeights[0] + minNeighWeight / 2)
                if q != 0:
                    cNeighWeights.append(bOnlyNeighWeights[0] + minNeighWeight / 2)
                if r != 0:
                    cNeighWeights.append(min(aShareNeighWeights[0], bShareNeighWeights[0]) + minNeighWeight / 2)
        return cNeighWeights
