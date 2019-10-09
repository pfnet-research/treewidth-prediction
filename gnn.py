#!/usr/bin/env python
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import pickle

import matplotlib
import numpy as np

import util
matplotlib.use('Agg')


class MLP(chainer.ChainList):
    """
    A class for Multi Layer Perceptron (MLP) which learns
    Aggregation and Combinine funtion.
    This MLP infer input dimension, so you don't have to set input dimension.
    And this MLP has BatchNormalization layer.

    Attributes
    ----------
    n_layers : int
        number of layers in MLP
    n_hidden : int
        number of hidden units in each layer
    n_out : int
        number of output vector dimension
        i.e. number of classes for classification task
    """
    def __init__(self, n_layers, n_hidden, n_out):
        """
        Parameters
        ----------
        n_layers : int
            number of layers in MLP
        n_hidden : int
            number of hidden units in each layer
        n_out : int
            number of output vector dimensions
            i.e. number of classes for classification task
        """
        super(MLP, self).__init__()

        # define the layers
        for layer in range(n_layers):
            if layer == 0:
                fc = L.Linear(None, n_hidden)
            elif layer != n_layers - 1:
                fc = L.Linear(n_hidden, n_hidden)
            elif layer == n_layers - 1:
                fc = L.Linear(n_hidden, n_out)

            self.add_link(fc)
            fc.name = "fc{}".format(layer)

            # add normalization layer
            if layer != n_layers - 1:
                norm = L.BatchNormalization(n_hidden)
                self.add_link(norm)
                norm.name = "norm{}".format(layer)

    # define forward calculation
    def __call__(self, x):
        for link in self.children():
            pre_activate = link(x)
            if 'fc' in link.name:
                x = pre_activate
            elif 'norm' in link.name:
                x = F.relu(pre_activate)
        return pre_activate


class GNN(chainer.Chain):
    """
    A class for Graph Neural Network which can change
    aggregation operation and readout operation.

    Attributes
    ----------
    num_layers : int
        number of layers in GNN
        i.e. how many aggregate neighborhood feature vectors in GNN
    num_mlp_layers : int
        number of layers in MLP
    input_dim : int
        number of node feature vector dimensions
    hidden_dim : int
        number of hidden units in MLP
    output_dim : int
        number of output vector dimensions
        i.e. number of classes in classification task,
        number of graph feature vector dimensions in regression task
    final_dropout : float
        dropout ratio on the final linear layer
    graph_pooling_type : [sum, average, max]
        how to aggregate entire nodes in a graph
    neighbor_pooling_type : [sum, average, max]
        how to aggregate neighbors
    task_type : [BinaryClassification, MulticlassClassification, Regression]
        which task do you want GNN to learn.
        Each task is described in Research Blog.
    """

    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout,
                 graph_pooling_type, neighbor_pooling_type, task_type="Classification"):
        """
        Parameters
        ----------
        num_layers : int
            number of layers in GNN
            i.e. how many aggregate neighborhood feature vectors in GNN
        num_mlp_layers : int
            number of layers in MLP
        input_dim : int
            number of node feature vector dimensions
        hidden_dim : int
            number of hidden units in MLP
        output_dim : int
            number of output vector dimensions
            i.e. number of classes in classification task,
            number of graph feature vector dimensions in regression task
        final_dropout : float
            dropout ratio on the final linear layer
        graph_pooling_type : [sum, average, max]
            how to aggregate entire nodes in a graph
        neighbor_pooling_type : [sum, average, max]
            how to aggregate neighbors
        task_type : [BinaryClassification, MulticlassClassification, Regression]
            which task do you want GNN to learn.
            Each task is described in Research Blog.
        """
        super(GNN, self).__init__()
        self.final_dropout = final_dropout
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.task_type = task_type

        with self.init_scope():
            self.mlps = chainer.ChainList(*[MLP(num_mlp_layers, hidden_dim, hidden_dim) for layer in range(self.num_layers - 1)])  # list of MLPs
            self.batch_norms = chainer.ChainList(*[L.BatchNormalization(hidden_dim) for layer in range(self.num_layers - 1)])  # list of batchnorms applied to the output of MLP
            self.linears_prediction = chainer.ChainList(*[L.Linear(output_dim) for layer in range(num_layers)])

            # In regression task, graph feature vector is transformed by 2-Layers MLP
            if task_type == "Regression":
                self.final_l2 = L.Linear(hidden_dim)
                self.final_l1 = L.Linear(1)

    def __call__(self, batch_graph, targets=None):
        """
        This method performs forward calculation.

        Parameters
        ----------
        batch_graph : list consists of Graph
            contains Graphs in minibatch
        targets : targets
            this parameter is only used in regression task

        Returns
        -------
        In classification task : (batchsize, num_classes) matrix
            which means the probability of which class is each graph in.
        In regression task : (batchsize, 1) matrix
            which means the prediction value of each graph treewidth.
        """
        # set the array module based on using device
        xp = self.device.xp

        # concatenate the node_features
        X_concat = chainer.Variable(xp.concatenate([xp.array(graph.node_features) for graph in batch_graph], axis=0))
        X_concat.to_device(self.device)  # if you use GPU, you must transfer X_concat into GPU.

        # make graph pooling matrix and neighbors pooling matrix
        graph_pool = self.__preprocess_graphpool(batch_graph)
        if self.neighbor_pooling_type == "max":
            np.set_printoptions(threshold=np.inf)
            np.set_printoptions(linewidth=3000)
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        hidden_rep = [X_concat]  # list of hidden representation at each layer (including input feature vectors)
        h = X_concat

        # perform Aggregating and Combining node features
        for layer in range(self.num_layers-1):
            # perform max neighbor pooling
            if self.neighbor_pooling_type == "max":
                # padding minimum value vector
                padded_h = F.concat((h, F.min(h, axis=0).reshape(1, h.shape[1])), axis=0)

                # make (F-dim, max_deg * nodes) matrix to perform max aggregation
                pooled_mat = F.sparse_matmul(padded_h.transpose(), padded_neighbor_list).transpose()

                # make 3D tensor
                pooled_tensor = F.reshape(pooled_mat, (padded_neighbor_list.shape[0] - 1,
                                          int(padded_neighbor_list.shape[1] / (padded_neighbor_list.shape[0] - 1)), h.shape[1]))

                # take max
                pooled = F.max(pooled_tensor, axis=1)

            # perform sum or average neighbor pooling
            else:
                pooled = F.sparse_matmul(Adj_block, h)
                if self.neighbor_pooling_type == "average":
                    degree = F.sparse_matmul(Adj_block, xp.ones((Adj_block.shape[0], 1), dtype=xp.float32))
                    pooled = pooled/degree

            # input aggregated vectors into MLP
            pooled_rep = self.mlps[layer](pooled)
            h = self.batch_norms[layer](pooled_rep)
            h = F.relu(h)
            hidden_rep.append(h)

        # perform Readout node features
        score_over_layer = 0
        for layer, h in enumerate(hidden_rep):
            # perform max readout
            if self.graph_pooling_type == "max":
                # padding minimum value
                padded_h = F.concat((h, F.min(h, axis=0).reshape(1, h.shape[1])), axis=0)

                # make (F-dim, max|V| * batchsize) matrix to perform max aggregation
                pooled_mat = F.sparse_matmul(padded_h.transpose(), graph_pool).transpose()

                # make 3D tensor
                pooled_tensor = F.reshape(pooled_mat, (len(batch_graph), int(graph_pool.shape[1] / len(batch_graph)), h.shape[1]))

                # take max
                pooled_h = F.max(pooled_tensor, axis=1)

            # sum or average readout
            else:
                pooled_h = F.sparse_matmul(graph_pool, h)

            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout)

        # final layers in regression task
        if self.task_type == "Regression":
            h = self.final_l2(score_over_layer)
            h = F.relu(h)
            score_over_layer = self.final_l1(h)

            if targets is None:
                return score_over_layer
            else:
                self.loss = F.mean_squared_error(targets.reshape(-1, 1), score_over_layer)  # MSE Loss
                self.abs_loss = F.mean_absolute_error(targets.reshape(-1, 1), score_over_layer)  # MAE Loss
                self.abs_max_loss = F.max(F.absolute_error(targets.reshape(-1, 1), score_over_layer))  # Max Absolute Error
                chainer.reporter.report({'loss': self.loss}, self)
                chainer.reporter.report({'abs_loss': self.abs_loss}, self)
                chainer.reporter.report({'abs_max_loss': self.abs_max_loss}, self)
                # return the MSE loss. If you want to use other loss, please change this sentence.
                return self.loss

        return score_over_layer

    def __preprocess_neighbors_sumavepool(self, batch_graph):
        # create block diagonal sparse matrix
        xp = self.device.xp
        edge_mat_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])

        Adj_block_idx = xp.concatenate(edge_mat_list, axis=1)
        Adj_block_elem = xp.ones(Adj_block_idx.shape[1], dtype=xp.float32)

        num_node = start_idx[-1]
        self_loop_edge = xp.array([range(num_node), range(num_node)])

        elem = xp.ones(num_node, dtype=xp.float32)
        Adj_block_idx = xp.concatenate([Adj_block_idx, self_loop_edge], axis=1)
        Adj_block_elem = xp.concatenate((Adj_block_elem, elem), axis=0)
        Adj_block = chainer.utils.CooMatrix(Adj_block_elem, Adj_block_idx[0], Adj_block_idx[1], shape=(start_idx[-1], start_idx[-1]))
        return Adj_block

    def __preprocess_neighbors_maxpool(self, batch_graph):
        # create sparse matrix for max neighborhood aggregation in batch_graph
        xp = self.device.xp
        row = []
        col = []
        elem = []
        start_idx = [0]
        max_deg = 0
        num_nodes = 0

        # compute the maximum degree within the graphs in the current minibatch
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            max_deg = max(max_deg, graph.max_neighbor)

        # constructing pooling matrix
        for i, graph in enumerate(batch_graph):
            for j in range(len(graph.neighbors)):
                # enumerate j's neighor
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                row.extend(pad)
                col += list(range(num_nodes * max_deg, num_nodes * max_deg + len(pad)))

                # calculates how long padding minimum feature vector
                padding_length = max_deg - len(pad)

                if not padding_length == 0:
                    row.extend([start_idx[-1]] * padding_length)
                    col += list(range(len(pad) + num_nodes * max_deg, (num_nodes + 1) * max_deg))

                num_nodes += 1

        elem.extend([1.0]*len(row))
        elem = xp.array(elem, dtype=xp.float32)
        row = xp.array(row)
        col = xp.array(col)
        return chainer.utils.CooMatrix(elem, row, col, shape=(start_idx[-1] + 1, max_deg * start_idx[-1]))

    def __preprocess_graphpool(self, batch_graph):
        # create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        # or create matrix used in max graph pooling
        xp = self.device.xp
        start_idx = [0]
        max_V = 0
        row = []
        col = []
        elem = []
        idx = []

        # compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            max_V = max(max_V, len(graph.g))

        if self.graph_pooling_type == "max":
            # constructing zero padding matrix (nodes, Max|V| * batch_size)
            for i in range(0, len(start_idx) - 1):
                # Note: g_i has start_idx[i+1] - start_idx[i] vertices
                row += list(range(start_idx[i], start_idx[i + 1]))
                col += list(range(i * max_V, i * max_V + start_idx[i + 1] - start_idx[i]))

                # padding minimum feature vector
                padding_length = (max_V - start_idx[i + 1] + start_idx[i])
                if not padding_length == 0:
                    row.extend([start_idx[-1]] * padding_length)
                    col += list(range(i * max_V + start_idx[i + 1] - start_idx[i], ((i + 1) * max_V)))

            # making the row list, col list and elem list
            elem.extend([1.0] * len(row))
            elem = xp.array(elem, dtype=xp.float32)
            row = xp.array(row)
            col = xp.array(col)

            return chainer.utils.CooMatrix(elem, row, col, shape=(start_idx[-1] + 1, max_V * len(batch_graph)))

        for i, graph in enumerate(batch_graph):
            if self.graph_pooling_type == "average":
                # for average pooling
                elem.extend([1.0 / len(graph.g)] * len(graph.g))
            else:
                # for sum pooling
                elem.extend([1.0]*len(graph.g))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])

        elem = xp.array(elem, dtype=xp.float32)
        idx = xp.array(idx).transpose()
        graph_pool = chainer.utils.CooMatrix(elem, idx[0], idx[1], shape=(len(batch_graph), start_idx[-1]))

        return graph_pool


def main(args):
    device = chainer.get_device(args.device)

    print('Training for {}'.format(args.task_type))
    print('Using Device: {}'.format(device))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Aggregation Type: {0}, Readout Type: {1}'.format(
          args.neighbor_pooling_type, args.graph_pooling_type))
    print('')

    # Load Grapdata
    dataset = util.GraphData(args.dataset, args.feat_init_method,
                             args.task_type, args.dataset_num, device)

    # Setup a model
    if args.task_type == "Regression":
        # For Regression
        model = GNN(args.num_layers, args.num_mlp_layers,
                    dataset.graphs[0].node_features.shape[1], args.hidden_dim,
                    dataset.graphs[0].node_features.shape[1], args.final_dropout,
                    args.graph_pooling_type, args.neighbor_pooling_type,
                    args.task_type)
    else:
        # For Classification
        model = L.Classifier(GNN(args.num_layers, args.num_mlp_layers,
                                 dataset.graphs[0].node_features.shape[1],
                                 args.hidden_dim, dataset.num_classes,
                                 args.final_dropout, args.graph_pooling_type,
                                 args.neighbor_pooling_type, args.task_type))

    # Choose the using device
    model.to_device(device)
    device.use()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Split the dataset into traindata and testdata
    train, test = chainer.datasets.split_dataset_random(dataset, int(dataset.__len__() * 0.9))
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set the output file name and path
    agg_name = args.neighbor_pooling_type + "_" + args.graph_pooling_type
    res_path = args.task_type

    # Set up a trainer
    updater = training.updaters.StandardUpdater(train_iter, optimizer,
                                                device=device, converter=dataset.converter)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out + "/" + res_path)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=device, converter=dataset.converter))

    trainer.extend(extensions.LogReport(filename='log_{}.dat'.format(agg_name)))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                         'epoch', file_name='loss_{}.png'.format(agg_name)))

    if args.task_type == "Regression":
        trainer.extend(extensions.PlotReport(['main/abs_loss', 'validation/main/abs_loss'],
                                             'epoch', file_name='absloss_{}.png'.format(agg_name)))
        trainer.extend(extensions.PlotReport(['main/abs_max_loss', 'validation/main/abs_max_loss'],
                                             'epoch', file_name='max_absloss_{}.png'.format(agg_name)))

        # Print selected entries of the log to stdout
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                                               'main/abs_loss', 'validation/main/abs_loss', 'main/abs_max_loss',
                                               'validation/main/abs_max_loss', 'elapsed_time']))

    else:
        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy_{}.png'.format(agg_name)))

        # Print selected entries of the log to stdout
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
                                               'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()

    # Save the trained model and model attributes
    if args.save_model:
        chainer.serializers.save_npz('./result/{0}/{1}.model'.format(res_path, agg_name), model)
        # save the model parameters (i.e. args)
        with open('./result/{0}/{1}.model_stat'.format(res_path, agg_name), 'wb') as f:
            pickle.dump(args, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The implementation of'
                                                 'general framework for GNN')
    parser.add_argument('--dataset', type=str, default="Erdos_Renyi",
                        help='name of random graph distribution dataset (default: Erdos_Renyi)')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of data in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=350,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--feat_init_method', type=str, default="binary",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--autoload', action='store_true',
                        help='Automatically load trainer snapshots in case'
                        ' of preemption or other temporary system failure')
    parser.add_argument('--save_model', action='store_true',
                        help='save the trained model')
    parser.add_argument('--model_name', default='ER_EP100_nsum_gsum.model', help='choose using trained model')
    parser.add_argument('--task_type', type=str, default='Task1', help='Choose task types Binary Classification, MultiClass Classification, Regression...')
    parser.add_argument('--dataset_num', type=int, default=2000, help='the number of dataset')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device', type=int, nargs='?', const=0, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    main(args)
