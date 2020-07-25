class ParaSettings:
    def __init__(self,
                 p=None,
                 k=10,
                 max_layer=None,
                 alpha=0.1,
                 num_buckets=None,
                 normalize=True,
                 gamma_struc=1,
                 gamma_attr=1):
        self.p = p  # num of landmark points
        self.k = k  # parameter to control landmark size
        self.max_layer = max_layer  # furthest hop distance up to which to compare neighbors
        self.alpha = alpha  # discount factor for higher layers
        self.num_buckets = num_buckets  # number of buckets to split node feature into currently base of log scale
        self.normalize = normalize  # whether to normalize node embeddings
        self.gamma_struc = gamma_struc  # parameter weighing structural similarity in node identity
        self.gamma_attr = gamma_attr  # parameter weighing attribute similarity in node identity


class Graph:
    """
    Discontinuous id of nodes in the graph are reconstructed in continuous index from 0 to maximum num of nodes in
    adjacency matrix.
    """

    def __init__(self,
                 adj,
                 all_nodes_num=None,
                 ori_nodes_num=None,
                 ori_nodes=None,
                 all_nodes=None,
                 node_labels=None,
                 edge_labels=None,
                 graph_label=None,
                 node_attributes=None):
        self.G_adj = adj  # adjacency matrix
        self.ori_nodes = ori_nodes  # original nodes in graph have been not compressed
        self.ori_nodes_num = ori_nodes_num  # number of original nodes in graph have been not compressed
        self.all_nodes = all_nodes  # all nodes in graph, including ori uncompressed and sup nodes
        self.all_nodes_num = all_nodes_num  # number of all nodes
        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.graph_label = graph_label
        self.node_attributes = node_attributes  # N x A matrix, where N is # of nodes, and A is # of attributes
