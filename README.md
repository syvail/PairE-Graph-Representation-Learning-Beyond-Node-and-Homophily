# PairE-Graph-Representation-Learning-Beyond-Node-and-Homophily
"Graph Representation Learning Beyond Node and Homophily"
# PairE: Graph Representation Learning Beyond Node and Homophily

We provide the source code of paire. Our model implementation is based on keras, which allows the model to be trained by GPU.

## Usage

#### Installation

- run the following code
    ```bash
    pip install -r requirements.txt
    ```
The source code is saved in the pair-embedding.ipynb file


#### Example

To run "PairE" on Cora network and evaluate the learned representations on multi-label node classification task, run the following command in the home directory of this project:

#### Input
The supported input format is an edgelist or an adjlist:
```text
edgelist: node1 node2 
adjlist: node n1 n2 n3 ... nk
```
The graph is assumed to be undirected and unweighted by default. 
The model needs additional features, the supported feature input format is as follow (**feature_i** should be a float number):

```text
node feature_1 feature_2 ... feature_n
```
##### Read data:
```python
dataset = 'pubmed'
iG,G,G_label,G_attr = read_data(dataset)
num_classes = G_label['label'].map(lambda x:x[0]).nunique()
```

#### Output
The output contains two dataframes: pair_embedding has *|E|* lines for a graph with *|E|* edges. Node_embedding has *n* lines for a graph with *n* nodes.
The  *|E|* lines are as follows:

```text
Pair<u, v> dim1 dim2 … dimd
```
The *n* lines are as follows:
```text
node_id dim1 dim2 ... dimd
```

where dim1, ... , dimd is the *d*-dimensional representation learned by *PairE*.
##### Train model :

```python
model = PairEmbedding(G,G_attr,G_label,embedding_size=128,epochs=30,silent=False)
node_embedding, edge_embedding, MultiTask_SelfSupervision_AE = model.fit()
```

#### Evaluation

If you want to evaluate the learned node representations, you can input the node labels. It will use a portion of nodes(default:10%、30%、50%、70%、90%) to train a classifier and calculate F1-score on the rest dataset.

The supported input label format is

    node label1 label2 label3...
##### evaluate on multi-class node classfication:
```python
model.evaluate(clf=[LogisticRegression(n_jobs=-1)])
```
#### Embedding visualization

We apply the dimensionality reduction method like t-SNE to the embedded visualization, and visualize the embedding of different data sets, where the colors of nodes represent the labels of nodes.
```python
plot_embedding(node_embedding, G_label, dataset)
```

