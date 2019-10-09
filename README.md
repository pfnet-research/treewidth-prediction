# Prediction of Treewidth using Graph Neural Network
This code predicts the treewidth by using graph neural network.
If you want more information about this code, please read my article ["Graph Neural Network を用いたグラフの木幅予測"](https://research.preferred.jp/2019/10/treewidth-prediction/) (sorry, written in Japanese) on [Preferred Networks Research Blog](https://research.preferred.jp).

This code also implements a general framework of GNN described in [1].
In detail, this code implements max, average and sum aggregations and readouts described the paper and is compatible with GPU.
We also provide the trained model and random graph datasets.

This software is developed as part of [PFN summer internship 2019](https://preferred.jp/en/news/internship2019/) and the main developer is [Yuta Nakano](https://github.com/mits58).

## Outline of prediction flow
This code follows the prediction flow described below.
1. Training GNN in classification task or regression task.
2. Predicting the treewidth of a given graph by using trained GNN model.

Details are written in above-mentioned article.

## Dependencies
This code has been tested over Chainer 6.3.0 and python 3.7.4.
If you haven't installed chainer, please install chainer following the instructions on the [official website](https://docs.chainer.org/en/stable/install.html).
If you want to run on GPU, please install CuPy following the instructions on the [official website](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy).

Here is other dependencies.
- matplotlib
- networkx
- numpy
- scikit-learn
- scipy
- seaborn
- timeout-decorator

You can install these dependencies by requirements.txt. Please run the below script.
```
pip install -r requirements.txt
```

## Example
### Training GNN (You can skip this)
Note: We provide pretrained model, so you can use this model and skip this step.

First, unzip the dataset file by running the below script.
```
./unpack.sh
```

#### Approach 1
Run the below script.
```
python GNN.py --task_type Task1
```

Default parameters are not the best performing hyper-parameters.
So you may want to tune hyper-parameters.
To check hyper-parameters to be specified, please type below.
```
python GNN.py --help
```

#### Approach 2
Run the below script.
```
python GNN.py --task_type Regression
```


### Predicting Treewidth

#### Approach 1
Run the below script.
```
python main.py --task_type approach1
```

#### Approach 2
Run the below script.
```
python main.py --task_type approach2
```

## License
MIT License. We provide no warranty or support for this implementation. Use it at your own risk.

Please see the [LICENSE](LICENSE) file for details.

## Reference
1. Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka. How Powerful are Graph Neural Networks?, ICLR 2019. [arXiv](https://arxiv.org/abs/1810.00826)
2. PACE-challenge/Treewidth: List of Treewidth solvers, instances, and tools, [https://github.com/PACE-challenge/Treewidth](https://github.com/PACE-challenge/Treewidth)
