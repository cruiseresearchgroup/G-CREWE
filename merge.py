class MERGE:
    """
    Compress a graph by merging a node with its neighbours that have a same minimum degree.
    """

    def __init__(self, graph, args, ini_super_node_id=0, node_num=0, nodes_list_comp=None):
        self.graph = graph  # a networkx graph
        self.guide_list = nodes_list_comp  # list of the nodes id for compression
        self.super_nodes_dict = {}  # a dictionary to save super-nodes and their sub-nodes
        self.ini_super_node_id = ini_super_node_id
        self.ori_node_num = node_num
        self.min_neigh_threshold = 0
        self.merge_mode = args.merge_mode
        self.a_node_select_mode = args.a_node_select_mode
        self.compress_ratio = args.varphi

    def execute(self):
        comp_nodes_num = int(self.compress_ratio * self.ori_node_num)
        merge_node_id = -1
        if self.a_node_select_mode == 0:
            while comp_nodes_num > 0:
                merge_node_id += 1
                try:
                    a_id = self.guide_list[merge_node_id]
                except IndexError:
                    merge_node_id = -1
                    continue

                success, nodes_num = self.merge(a_id)
                if success:
                    comp_nodes_num -= nodes_num
                    # print("comp_nodes_num:", comp_nodes_num)
                else:
                    print("merge fail a_id " + str(a_id))

    def merge(self, aId):
        if not self.graph.has_node(aId):  # if not - it is zero, empty or False
            return False, 0

        a_neighbours = list(self.graph.neighbors(aId))  # id of all neighbours of node a
        if not len(a_neighbours):  # if it is empty
            return False, 0

        # find A neighbours that have minimum degree
        a_neigh_degrees = list(self.graph.degree(a_neighbours))
        a_neigh_degrees = sorted(a_neigh_degrees, key=lambda tup: tup[1])  # sort by degree
        a_neighbours_min = [n_id for (n_id, y) in a_neigh_degrees if y == a_neigh_degrees[0][1]]

        # collect all the neighbours of the nodes in a_neighbours_min and A itself, with set
        all_neighbours_to_conn = set(a_neighbours)
        for n_id in a_neighbours_min:
            neighbours = list(self.graph.neighbors(n_id))  # id of all neighbours of node b
            all_neighbours_to_conn.update(neighbours)

        a_neighbours_min.append(aId)  # the nodes will be compressed into a super-node
        super_node_id = -1  # the node id as the super node after compression
        all_sub_nodes = set([])  # the original nodes in each super-node that will be compressed together
        for n_id in a_neighbours_min:
            all_neighbours_to_conn.discard(n_id)  # remove the nodes that will be compressed into a super-node
            if n_id in self.super_nodes_dict:  # if the node is a super-node, add its sub-nodes
                all_sub_nodes.update(self.super_nodes_dict[n_id])
                if super_node_id < n_id:  # select the biggest one for super-node id
                    super_node_id = n_id
            else:  # add when it is an original node
                all_sub_nodes.update([n_id])

        if super_node_id == -1:  # if there is no super nodes in future compression, give a new id
            super_node_id = self.ini_super_node_id
            self.ini_super_node_id += 1
        # add or update a super node from the list
        self.super_nodes_dict[super_node_id] = all_sub_nodes

        # remove the nodes that will be compressed
        for n_id in a_neighbours_min:
            self.graph.remove_node(n_id)  # remove a node from networkx graph
            if n_id in self.super_nodes_dict and super_node_id != n_id:
                self.super_nodes_dict.pop(n_id)  # delete other super nodes from storage list
            if n_id in self.guide_list:
                self.guide_list.remove(n_id)

        # add the super node to networkx graph
        self.graph.add_node(super_node_id)
        self.guide_list.append(super_node_id)

        # add the new edges and weights for super-nodes' connections
        default_weight = 1
        for n_id in all_neighbours_to_conn:
            self.graph.add_edge(super_node_id, n_id, weight=default_weight)

        num_comp_nodes = len(a_neighbours_min) - 1
        return True, num_comp_nodes

    def update_threshold(self, diff):
        self.min_neigh_threshold = self.min_neigh_threshold + diff

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
