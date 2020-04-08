# WoMG: Word of Mouth Generator
*WoMG* is a Python library for **Word-of-Mouth** Cascades **Generation**.

WoMG is a syntectic data generator which combines  topic  modeling  and  a  topic-aware  propagation model  to  create  realistic  information-rich  cascades,  whose shape depends on many factors, including the topic of theitem and its virality,  the homophily of the social network, the interests of its users and their social influence.

*WoMG* generates synthetic datasets of documents cascades on network.
It starts with any (un)directed, (un)weighted graph and a collection of documents and it outputs the propagation DAGs (Directed-Acyclic graph) of the documents through the network.

<img src="data/images/womg.png" height="480px" />


## Installation

### Dependencies

### User installation
Install using ``pip``: <br>

```bash
$ pip install womg
```

#### Source code
You can also download or clone the GitHub repository: <br>

```bash
$ git clone https://github.com/FedericoCinus/WoMG.git
```


## Usage
The WoMG package provides a Python module and a command-line method. To import the womg function type:<br/>

```python
from womg import womg
```
The **demo.ipynb** provides a tutorial.


#### Options
You can check out the other options available to use with *WoMG* using:<br/>

```python
?womg
```

0. ``topics`` number of topics to be considered in the topic distributions of documents and nodes interests; it has to be less than number of dimensions of the nodes' space provided by node2vec

#### Input
[Network] The supported input format are: NetworkX instance and edgelist (txt extension):

		node1_id_int node2_id_int <weight_float, optional>

You can specify the edgelist path using the *graph* argument:

```bash
womg(graph='/this/is/an/example/path/Graph_Folder/edgelist.txt')
```

#### Output (default)
1. [Propagations] The output format is:

		time; item; node
2. [Items descriptions] :

		item; [topic-dim vector]

3. [Topic descriptions] :

		(topic_index, linear combination of words)

All outputs are returned by the **womg** function and saved in the current directory.

You can also specify the output folder path:

```python
womg(path_out='/this/is/an/example/path/Output_Folder')
```

## Authors
This repository provides a reference implementation of *WoMG* as described in:<br>

> WoMG: a Library for Word-of-Mouth Cascades Generation.<br>
> Federico Cinus, Francesco Bonchi, Corrado Monti, André Panisson.<br>
> <Insert paper link>
	
	
> Generating realistic interest-driven information cascades.<br>
> Federico Cinus, Francesco Bonchi, Corrado Monti, André Panisson.<br>
> <Insert paper link>
