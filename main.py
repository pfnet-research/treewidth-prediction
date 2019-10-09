import argparse
import time
import timeout_decorator
import pickle

import chainer
import chainer.functions as F
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
import seaborn as sns

import util
from gnn import GNN


def int_set(subset):
    """
    This function transforms subset into corresponding integer representation.

    Parameters
    ----------
    subset : set
        The subset you want to transform.

    Returns
    -------
    representation : int
        The integer representation of a given subset.
    """
    representation = 0
    for i in subset:
        representation += (1 << i)

    return representation


def set_str(subset, v):
    """
    This function transforms a subset and a vertex into corresponding string representation.

    Parameters
    ----------
    subset : set
        The subset you want to transform.
    v : int
        The vertex you want to transform.

    Returns
    -------
    representation : string
        The string representation of a given subset and a given vertex.
    """
    return str(int_set(subset)) + " " + str(v)


def Q(G, S, v):
    """
    This function calculates Q(S, v) = {w ∈ V − S − {v} | there is a path from v to w in G[S ∪ {v, w}]}.
    The time complexity of this function is O(n + m) time.

    Parameters
    ----------
    G : Graph object
        entire graph
    S : set
        a subset of vertices
    v : int
        vertex

    Returns
    -------
    {w ∈ V − S − {v} | there is a path from v to w in G[S ∪ {v, w}]} : int
        The string representation of a given subset and a given vertex.
    """
    cnt = 0
    induce_V = S
    induce_V.add(v)

    # initialize union find tree
    uf = UnionFind(len(G.nodes))

    # calc connected component
    for edge in G.edges:
        (source, sink) = edge
        if source in induce_V and sink in induce_V:
            uf.Unite(source, sink)

    # enum paths
    for vertex in (set(G.nodes) - S - set([v])):
        # v and w is same connected component?
        for neighbor in G[vertex]:
            if uf.isSameGroup(v, neighbor):
                cnt += 1
                break

    return cnt


class UnionFind():
    '''
        The implementation of Disjoint-set data structure(as known as Union-Find tree)
    '''
    def __init__(self, n):
        self.n = n
        self.root = [-1]*(n+1)
        self.rnk = [0]*(n+1)

    def Find_Root(self, x):
        if(self.root[x] < 0):
            return x
        else:
            self.root[x] = self.Find_Root(self.root[x])
            return self.root[x]

    def Unite(self, x, y):
        x = self.Find_Root(x)
        y = self.Find_Root(y)
        if(x == y):
            return
        elif(self.rnk[x] > self.rnk[y]):
            self.root[x] += self.root[y]
            self.root[y] = x

        else:
            self.root[y] += self.root[x]
            self.root[x] = y
            if(self.rnk[x] == self.rnk[y]):
                self.rnk[y] += 1

    def isSameGroup(self, x, y):
        return self.Find_Root(x) == self.Find_Root(y)


class TreewidthAlgorithm():
    """
    A class for algorithms calculates treewidth.
    The attributes in this class have some

    Attributes
    ----------
    dp_S : set
        memorize the result of TWCheck(G, S)
    dp_Q : set
        memorize the result of Q(G, S, v)
    bound : string
        use upper-prune or lower-prune or both
    prune_num : int
        memorize the number of pruned search state
    func_call_num : int
        memorize the number of function calls
    eval_GNN_time : float
        memorize the time of evaluation GNN
    """
    def __init__(self, prune):
        self.dp_S = {}
        self.dp_Q = {}
        self.prune = prune
        self.prune_num = 0
        self.func_call_num = 0
        self.eval_GNN_time = 0

    def initialize(self):
        """
        This method is called in each opt to initialize dp_S
        """
        if self.prune == "upper":
            # preserve true
            new_S = {}
            for k, v in self.dp_S.items():
                if v:
                    new_S[k] = v
            self.dp_S = new_S

        elif self.prune == "lower":
            # preserve false
            new_S = {}
            for k, v in self.dp_S.items():
                if not v:
                    new_S[k] = v
            self.dp_S = new_S
        else:
            # preserve true
            new_S = {}
            for k, v in self.dp_S.items():
                if v:
                    new_S[k] = v
            self.dp_S = new_S

    def Q(self, G, S, v):
        '''
        This method calculates Q(S, v) = {w ∈ V − S − {v} | there is a path from v to w in G[S ∪ {v, w}]}.
        The time complexity of this function is O(n + m) time. This method uses dp_Q.

        Parameters
        ----------
        G : Graph object
            entire graph
        S : set
            a subset of vertices
        v : int
            vertex

        Returns
        -------
        {w ∈ V − S − {v} | there is a path from v to w in G[S ∪ {v, w}]} : int
            The string representation of a given subset and a given vertex.
        '''
        if set_str(S, v) in self.dp_Q:
            return self.dp_Q[set_str(S, v)]

        cnt = 0
        induce_V = S
        induce_V.add(v)

        # initialize union find tree
        uf = UnionFind(len(G.nodes))

        # calc connected component
        for edge in G.edges:
            (source, sink) = edge
            if source in induce_V and sink in induce_V:
                uf.Unite(source, sink)

        # enum paths
        for vertex in (set(G.nodes) - S - set([v])):
            # v and w is same connected component?
            for neighbor in G[vertex]:
                if uf.isSameGroup(v, neighbor):
                    cnt += 1
                    break

        self.dp_Q[set_str(S, v)] = cnt
        return cnt

    def calc_treewidth_recursive(self, G, S, opt):
        """
        This method judges whether G[S] has treewidth at most opt by using recursive algorithm.
        This method uses dp_S to reduce computational complexity.

        Parameters
        ----------
        G : Graph object
            entire graph
        S : set
            a subset of vertices
        opt : int
            a value to judge

        Returns
        -------
        Result : boolean
            whether the given graph has treewidth at most opt
        """
        self.func_call_num += 1

        if len(S) == 0:
            return True

        if len(S) == 1:
            return self.Q(G, set(), S.pop()) <= opt

        if int_set(S) in self.dp_S:
            return self.dp_S[int_set(S)]

        res = False

        for vertex in S:
            Qval_check = self.Q(G, S - set([vertex]), vertex) <= opt
            if Qval_check:
                res = (res or self.calc_treewidth_recursive(G, S - set([vertex]), opt))
            if res:
                break

        self.dp_S[int_set(S)] = res
        return res

    def calc_treewidth_with_GNN(self, G, S, opt, model, prob_bound):
        """
        This method judges whether G[S] has treewidth at most opt by using recursive algorithm.
        This method uses dp_S to reduce computational complexity.
        And this method also uses GNN to reduce the number of function calls.

        Parameters
        ----------
        G : Graph object
            entire graph
        S : set
            a subset of vertices
        opt : int
            a value to judge
        model : GNN object
            GNN model used in this method
        prob_bound : float
            The threshold of probability used in GNN-pruning

        Returns
        -------
        Result : boolean
            whether the given graph has treewidth at most opt
        """
        self.func_call_num += 1

        if len(S) == 0:
            return True

        if len(S) == 1:
            return self.Q(G.g, set(), S.pop()) <= opt

        if int_set(S) in self.dp_S:
            return self.dp_S[int_set(S)]

        # check whether we should branch from S using evaluation of tw(G[S])
        ev_st = time.time()
        # make the induced graph
        tmp_G = nx.relabel.convert_node_labels_to_integers(G.g.subgraph(S))
        induce_G = util.GraphData.nx_to_graph(tmp_G, "binary")

        if induce_G.g.number_of_edges() > 1:
            # predict the treewidth
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                pred = model.predictor([induce_G])
                prob = F.max(F.softmax(pred), axis=1).array[0]
                prediction = F.argmax(F.softmax(pred), axis=1).array[0]

            if prob > prob_bound:
                # use the predition
                if prediction == 0 and 2 * opt < induce_G.g.number_of_nodes() and (self.prune == "upper" or self.prune == "both"):
                    # upper-prune
                    self.prune_num += 1
                    self.dp_S[int_set(S)] = False
                    self.eval_GNN_time += time.time() - ev_st
                    return False
                if prediction == 1 and 2 * opt > induce_G.g.number_of_nodes() and (self.prune == "lower" or self.prune == "both"):
                    # lower-prune
                    self.prune_num += 1
                    self.dp_S[int_set(S)] = True
                    self.eval_GNN_time += time.time() - ev_st
                    return True

        # memorize GNN evaluation time
        self.eval_GNN_time += time.time() - ev_st

        # branch from this state
        res = False
        for vertex in S:
            Qval_check = self.Q(G.g, S - set([vertex]), vertex) <= opt
            if Qval_check:
                res = (res or (self.calc_treewidth_with_GNN(G, S - set([vertex]), opt, model, prob_bound)))
            if res:
                break

        self.dp_S[int_set(S)] = res
        return res


@timeout_decorator.timeout(600)
def test(eval_graph, model, prob_bound, prune):
    calc_GNN_tw = TreewidthAlgorithm(prune)

    start = time.time()
    if prune == "lower":
        # calculate lower bound by Lower-Prune with Upper-Calculation
        for opt in range(eval_graph.g.number_of_nodes() - 1, 0, -1):
            GNN_tw = calc_GNN_tw.calc_treewidth_with_GNN(eval_graph, eval_graph.g.nodes, opt, model, prob_bound)
            if not GNN_tw:
                break
            calc_GNN_tw.initialize()

    elif prune == "upper":
        # calculate upper bound by Upper-Prune with Lower-Calculation
        for opt in range(1, eval_graph.g.number_of_nodes()):
            GNN_tw = calc_GNN_tw.calc_treewidth_with_GNN(eval_graph, eval_graph.g.nodes, opt, model, prob_bound)
            if GNN_tw:
                break
            calc_GNN_tw.initialize()

    else:
        # calculate predicional treewidth by Both Pruning with Lower-Calculation
        for opt in range(1, eval_graph.g.number_of_nodes()):
            GNN_tw = calc_GNN_tw.calc_treewidth_with_GNN(eval_graph, eval_graph.g.nodes, opt, model, prob_bound)
            if GNN_tw:
                break
            calc_GNN_tw.initialize()

    end = time.time()
    return (end - start), opt + (1 if prune == "lower" else 0), (calc_GNN_tw.prune_num), calc_GNN_tw.func_call_num, calc_GNN_tw.eval_GNN_time


@timeout_decorator.timeout(600)
def ordinaryDP(eval_graph):
    calc_DP_tw = TreewidthAlgorithm("existAlg")
    start = time.time()
    for opt in range(1, eval_graph.g.number_of_nodes()):
        DP_tw = calc_DP_tw.calc_treewidth_DP_recursive(eval_graph.g, eval_graph.g.nodes, opt)
        if DP_tw:
            break
        calc_DP_tw.initialize()
    end = time.time()
    return end - start, opt, calc_DP_tw.func_call_num


def approach_1(args):
    device = chainer.get_device(args.device)

    # load the trained model, model attributes and graph dataset
    with open('{}_stat'.format(args.model_name), mode='rb') as f:
        model_args = pickle.load(f)

    dataset = util.GraphData("mixed", model_args.feat_init_method, "Regression", args.data_num, device)

    model = chainer.links.Classifier(GNN(model_args.num_layers, model_args.num_mlp_layers, dataset.graphs[0].node_features.shape[1],
                                         model_args.hidden_dim, 2, model_args.final_dropout, model_args.graph_pooling_type,
                                         model_args.neighbor_pooling_type, model_args.task_type))
    chainer.serializers.load_npz(args.model_name, model)

    model.to_device(device)
    device.use()

    print('\n--- Prediction by approach1 ---')
    print('Prob_bound\tPrune\tIndex\t|V|\t|E|\ttw(G)\ttime\tevaltw\tprunenum\tfunccallnum\tevalGNNtime')
    # calculation the treewidth of given graphs and evaluation the proposed method and ordinary DP
    prob_bounds = [0.5, 0.7, 0.9]   # using this bound when deciding whether using the prediction of GNN
    bounds = ["upper", "lower", "both"]     # using this bound at pruning
    for prob_bound in prob_bounds:
        for bound in bounds:
            output_file = bound + "_{0:02}_{1}.dat".format(int(prob_bound * 10), args.model_name)
            result = ["ID\t|V|\t|E|\ttw\ttime\tevaltw\tprunenum\tfunccallnum\tevalGNNtime"]

            for idx in range(0, len(dataset.graphs)):
                print('{0}\t{1}\t{2}'.format(prob_bound, bound, idx), end='\t')
                eval_graph = dataset.graphs[idx]
                graphstat = "{3}\t{0}\t{1}\t{2}".format(eval_graph.g.number_of_nodes(), eval_graph.g.number_of_edges(), dataset.labels[idx], str(idx).rjust(5))
                if eval_graph.g.number_of_nodes() > 15:
                    print()
                    continue
                try:
                    tm, evtw, pn, fcn, evtime = test(eval_graph, model, prob_bound, bound)
                    res = "{0}\t{1}\t{2}\t{3}\t{4}".format(tm, evtw, pn, fcn, evtime)
                except TimeoutError:
                    res = "TimeOut"
                print(graphstat + "\t" + res)
                result.append(graphstat + "\t" + res)
            # write results to a file
            if args.out_to_file:
                with open("./{}/Approach1/".format(args.out) + output_file, "w") as f:
                    f.write('\n'.join(result))

    print('\n--- Prediction by existing algorithm ---')
    print('Index\t|V|\t|E|\ttw(G)\ttime\tevaltw\tprunenum\tfunccallnum\tevalGNNtime')
    output_file = "exist_{0}.dat".format(args.model_name)
    result = ["ID\t|V|\t|E|\ttw\ttime\tevaltw\tfunccallnum"]

    for idx in range(0, len(dataset.graphs)):
        eval_graph = dataset.graphs[idx]
        graphstat = "{3}\t{0}\t{1}\t{2}".format(eval_graph.g.number_of_nodes(), eval_graph.g.number_of_edges(), dataset.labels[idx], str(idx).rjust(5))
        print('{0}\t{1}\t{2}\t{3}'.format(str(idx).rjust(5), eval_graph.g.number_of_nodes(), eval_graph.g.number_of_edges(), dataset.labels[idx]), end='\t')
        try:
            tm, evtw, fcn = ordinaryDP(eval_graph)
            res = "{0}\t{1}\t{2}".format(tm, evtw, fcn)
        except TimeoutError:
            res = "TimeOut"
        print(res)
        result.append(graphstat + "\t" + res)

    # write results to a file
    if args.out_to_file:
        with open("./{}/Approach1/".format(args.out) + output_file, "w") as f:
            f.write('\n'.join(result))


def approach_2(args):
    if args.model_name == './pretrained_models/sum_sum_binary.model':
        # change the default model for regression
        args.model_name = "./pretrained_models/sum_sum_regression.model"

    device = chainer.get_device(args.device)

    # load the trained model, model attributes and graph dataset
    with open('{}_stat'.format(args.model_name), mode='rb') as f:
        model_args = pickle.load(f)

    dataset = util.GraphData("mixed", model_args.feat_init_method, "Regression", args.data_num, device)
    # convert tuple dataset to graphs and targets
    graphs, targets = dataset.converter(dataset, device)

    model = GNN(model_args.num_layers, model_args.num_mlp_layers, dataset.graphs[0].node_features.shape[1],
                model_args.hidden_dim, dataset.graphs[0].node_features.shape[1], model_args.final_dropout,
                model_args.graph_pooling_type, model_args.neighbor_pooling_type, model_args.task_type)
    chainer.serializers.load_npz(args.model_name, model)

    model.to_device(device)
    device.use()

    print('\n--- Prediction by approach2 ---')

    # predict treewidth
    eval_st = time.time()
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        raw_output = model(graphs)
        prediction = np.round(raw_output.array).reshape(len(raw_output))
    eval_time = time.time() - eval_st

    # calculate some stats
    print('# of graphs\t{0}'.format(str(len(graphs)).rjust(5)))
    print('Execution time\t{0}'.format(eval_time))
    print('Execution time per each graphs\t{0}'.format(eval_time / len(graphs)))
    print('Mean Absolute Error\t{0}'.format(sklearn.metrics.mean_absolute_error(targets, prediction)))
    print('Max Error\t{0}'.format(sklearn.metrics.max_error(targets, prediction)))

    # output the scatter plot
    sns.set_style("whitegrid", {'grid.linestyle': '--'})

    df = pd.DataFrame({"Real Value": targets, "Predict Value": prediction})
    g = sns.jointplot(df["Real Value"], df["Predict Value"])
    g.ax_joint.plot([49, 1], [49, 1], ':k')
    # Please make this directory before run this code...
    plt.savefig('./{0}/Approach2/scatter.png'.format(args.out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction of Treewidth Using Graph Neural Network')
    parser.add_argument('--device', '-d', type=int, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--out_to_file', action='store_true',
                        help='output the results')
    parser.add_argument('--model_name', default='./pretrained_models/sum_sum_binary.model', help='choose trained model you want to use')
    parser.add_argument('--task_type', type=str, default='approach1', choices=['approach1', 'approach2'],
                        help='Which approach do you want to use')
    parser.add_argument('--data_num', type=int, default=100, help='the number of graphs you want to calculate treewidth')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device', type=int, nargs='?', const=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    if args.task_type == "approach1":
        approach_1(args)
    else:
        approach_2(args)
