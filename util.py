import collections
import pickle

import chainer
import numpy as np

np.random.seed(1124)


class Graph(object):
    """
    A class for graphs.

    Attributes
    ----------
    label : int
        the label of a graph
    treewidth : int
        the treewidth of a graph
    g : networkx Graph
        a graph object
    neighbors : list
        a adjacency list of the graph
    node_features :
        a matrix of node feature vectors
    edge_mat :
        a adjacency matrix of the graph in sparse format(coo format)
    max_neighbor :
        max degree of the graph
    """
    def __init__(self, g, label, tw, node_features=None):
        self.label = label
        self.treewidth = tw
        self.g = g
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0


class GraphData(chainer.dataset.DatasetMixin):
    """
    A class for graph data.

    Attributes
    ----------
    graphs : a list of Graph
        a set of Graph data
    num_classes : int
        the number of classes in classification task
    labels : int (in classification task) or float (in regression task)
        a set of labels
    task_type : string
        which task GNN will learn
    """

    def __init__(self, dataset, feat_init_method, task_type, dataset_num=None, device=-1):
        self.graphs, self.num_classes = GraphData.load_data(dataset, feat_init_method, task_type, dataset_num, device)
        self.task_type = task_type
        xp = device.xp
        if task_type == "Regression":
            self.labels = xp.array([g.treewidth for g in self.graphs], dtype=xp.float32)
        else:
            self.labels = xp.array([g.label for g in self.graphs], dtype=xp.int32)

    def __len__(self):
        return len(self.graphs)

    def get_example(self, i):
        return self.graphs[i], self.labels[i]

    def converter(self, datalist, device):
        xp = chainer.get_device(device).xp
        graphs = [data[0] for data in datalist]

        if self.task_type == "Regression":
            targets = xp.array([float(data[1]) for data in datalist], dtype=xp.float32)
        else:
            targets = xp.array([int(data[1]) for data in datalist], dtype=xp.int32)

        return graphs, targets

    def nx_to_graph(graph, feat_init_method):
        """
        This method transforms networkx graph object into Graph Object.

        Parameters
        ----------
        graph : networkx graph object
            the graph you want to transform
        feat_init_method : string
            how to initialize the node feature vectors

        Returns
        -------
        corresponding Graph Object
        """
        s2v_g = Graph(graph, -1, 0)

        s2v_g.neighbors = [[] for i in range(len(s2v_g.g))]
        for i, j in s2v_g.g.edges():
            s2v_g.neighbors[i].append(j)
            s2v_g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(s2v_g.g)):
            s2v_g.neighbors[i] = s2v_g.neighbors[i]
            degree_list.append(len(s2v_g.neighbors[i]))
        s2v_g.max_neighbor = max(degree_list)

        edges = [list(pair) for pair in s2v_g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        s2v_g.edge_mat = np.array(edges).transpose()

        # add feature vector
        if feat_init_method == "binary":
            # (|V|, 32) -> the binary representation of node's degree
            feat = []
            for node in s2v_g.g.nodes:
                bin_str = "{:032b}".format(s2v_g.g.degree(node))
                feat.append([float(w) for w in bin_str])
            s2v_g.node_features = np.array(feat, dtype=np.float32)
        else:
            # initialize [0] = 1 [other] = 0
            s2v_g.node_features = np.zeros((s2v_g.g.number_of_nodes(), int(feat_init_method)), dtype=np.float32)
            s2v_g.node_features[0] = 1

        return s2v_g

    def load_data(dataname, feat_init_method, task_type, dataset_num=1000, device=-1):
        """
        This method reads the serialized dataset in pickle.

        Parameters
        ----------
        dataname : string
            dataset name which you want to use
        feat_init_method : string
            how to initialize the node feature vectors
        task_type : string
            what task you want GNN to learn
        dataset_num : int (default : 1000)
            how many graphs you want to use
        device : Device object
            which device you want to use

        Returns
        -------
        graphs : list
            list of Graph class objects
        num_classes : int
            the number of classification classes
        """
        if device == -1:
            device = chainer.backends.cuda.get_device_from_id(device)
        xp = device.xp

        # read serialized dataset in pickle
        with open('./dataset/{}.pkl'.format(dataname), 'rb') as f:
            g_list = pickle.load(f)

        # attach labels to graphs
        cnt_labels = []
        for g in g_list:
            if task_type == "Task1":
                g.label = int(g.treewidth * 2 <= g.g.number_of_nodes())
            elif task_type == "Task2":
                # 5-class classification
                g.label = int(5.0 * g.treewidth / g.g.number_of_nodes())
            elif task_type.isdecimal():
                split_num = int(task_type)
                g.label = int(split_num * g.treewidth / g.g.number_of_nodes())
            else:
                # regression
                g.label = g.treewidth

            cnt_labels.append(g.label)

        # adjust the number of data and initialize node feature vectors
        res_g = []
        c = collections.Counter(cnt_labels)
        num = c.most_common()[-1][1]
        if dataset_num is not None:
            num = min(num, dataset_num)

        if task_type == "Regression":
            num = dataset_num

        dict_labels = {}
        for label in set(cnt_labels):
            dict_labels[label] = 0

        np.random.shuffle(g_list)

        for g in g_list:
            if not task_type == "Regression":
                if dict_labels[g.label] == num:
                    continue

                dict_labels[g.label] = dict_labels[g.label] + 1

            if task_type == "Regression":
                if num == len(res_g):
                    break

            if feat_init_method == "binary":
                # (|V|, 32) -> the 32-bit binary representation of node's degree
                feat = []
                for node in g.g.nodes:
                    bin_str = "{:032b}".format(g.g.degree(node))
                    feat.append([float(w) for w in bin_str])
                g.node_features = xp.array(feat, dtype=xp.float32)
            elif feat_init_method.isdecimal():
                # divide into int(feat_init_method) and one hot vector
                pass
            else:
                # initialize [0] = 1 [other] = 0, 10-dim vector
                g.node_features = xp.zeros((g.g.number_of_nodes(), 10), dtype=xp.float32)
                g.node_features[0] = 1

            # edge_mat to xp.array
            g.edge_mat = xp.array(g.edge_mat)

            res_g.append(g)

        print('--- {} dataset ---'.format(dataname))
        if task_type == "Regression":
            print('max(tw(G)): {0}, min(tw(G)): {1}'.format(max(cnt_labels), min(cnt_labels)))
        else:
            print('# classes: %d' % len(set(cnt_labels)))
        print('# feat. vector dim.: %d' % g_list[0].node_features.shape[1])
        print("# data: %d" % len(res_g))

        return res_g, len(set(cnt_labels))
