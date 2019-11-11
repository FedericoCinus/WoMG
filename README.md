# WoMG: Word of Mouth Generator
*WoMG* is a Python library for **Word-of-Mouth** Cascades **Generation**.

We propose a model for the synthetic generation of information cascades in social media. In our model the information “memes” propagating in the social network are characterized by a probability distribution in a topic space, accompanied by a textual description, i.e., a bag of keywords coherent with the topic distribution. Similarly, every person is described by a vector of interests defined over the same topic space. Information cascades are governed by the topic of the meme, its level of virality, the interests of each person, community pressure, and social influence.

This repository provides a reference implementation of *WoMG* as described in the:<br>
> Generating realistic interest-driven information cascades.<br>
> Federico Cinus, Francesco Bonchi, André Panisson, Corrado Monti.<br>
> <Insert paper link>

*WoMG* generates synthetic datasets of documents cascades on network. 
It starts with any (un)directed, (un)weighted graph and a collection of documents and it outputs the propagation DAGs of the docs through the network.

Immage


## Installation
Download or clone the GitHub repository. 
``git clone https://github.com/FedericoCinus/WoMG.git``

In order to install the core package, move to the *src/womg_core/* directory, which constains the *setup.py* file, and install typing:
``pip install -e .``



## Usage
The WoMG package provides a Python module and a command-line method.

### Quickstart
 To run *WoMG* on a **demo** mode, execute the following command from Terminal:<br/>

``womg``

It will generates 100 documents to be spread over the default network (*Les Miserables* http://konect.uni-koblenz.de/networks/moreno_lesmis).

### Options
You can check out the other options available to use with *WoMG* using:<br/>

``womg --help``

### Input
1. [Network] The supported input format is an edgelist (txt extension):
	
		node1_id_int node2_id_int <weight_float, optional>
		
You can specify the edgelist path using the *graph* argument:

``womg --graph /this/is/an/example/path/Graph_Folder/edgelist.txt``

If no path is given the default network is *Les Miserables* network.

2. [Documents] The supported input format for documents collection (corpus) is txt. You have to specify the folder path containing them using the *docs_folder* argument:

 ``womg --docs_folder /this/is/an/example/path/Corpus_Folder``
 
If no documents folder path is given, *WoMG* will be set to generative mode.
  
### Output
Diffusion output file is in a dict format: 

	time; item; node

You can specify the output folder path:

  `` womg --output /this/is/an/example/path/Output_Folder``




## WoMG extended (TBD)
*WoMG* is an open source reasearch project. More details of the software are reported below:

### Input
1. [Network] The supported input format is an edgelist (txt extension):
	
		node1_id_int node2_id_int <weight_float, optional>
		
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags. You can specify the edgelist path using the *graph* argument):

``python womg --graph /this/is/an/example/path/Graph_Folder/edgelist.txt``

If no path is given the default network is *Les Miserables* network.

2. [Documents] The supported input format for documents collection (corpus) is txt. You have to specify the folder path containing them using the *docs_folder* argument:

 ``python womg --docs_folder /this/is/an/example/path/Corpus_Folder``

If no documents folder path is given, WoMG will be set to generative mode.
  
### Output
There are outputs for each class (or model)

1. [Diffusion] file could be in two formats:

  list (default): 
  
	time doc activating_node
  dict : 
  
  	{ time: { doc: [activating nodes] } }

2. [Network] files:
  [info] dict: 
  
	{'type': 'Graph', 'numb_nodes': '77', 'numb_edges': '254', 'aver_degree': '6.5974', 'directed': 'False'} 
	
  [graph] dict: 
  
      {(u, v): [1.3, 0.2, 0.8, ... , 0.91], ...}
      
  Key: link-tuple. Value: weight vector

  [interests and influence vectors] dict:
  
	{(node, 'int'): [interest vector], (node, 'inlf'): [influence vector]}

3. [Topic] files:

  [topic distributions] dict:
  
      {doc: [topic distribution]}
      
  [viralities] dict:
  
      {doc: virality}

One can modify the outputs formats extension with the *format* argument:

  ``python womg --format pickle``
  ``python womg --format txt``
  
and specify the output folder path:

  ``python womg --output /this/is/an/example/path/Output_Folder``


### Options

0. ``topics`` number of topics to be considered in the topic distributions of documents and nodes interests; it has to be less than number of dimensions of the nodes' space provided by node2vec
##### Graph
1. ``homophily`` H degree of homophily. Node2vec is used as baseline for generating interests vectors of the nodes starting from the given graph. Parameters *p* and *q* can achieve different decoded degree of homophily and structural equivalence (see paper). The best mix of them can be achieved only by a deep analysis of the network and a grid searh on the parameters. In order to pursuit generality in the input graph we use three degree of mixing: structural equivalence predominant, deepWalk (p=1, q=1), homophily predominant (which are not the best for representing the graph!).  1-H is the degree of social influence between nodes; which is the percentage of the avg interests vecs norms to be assigned to the influence vectors.

##### Documents
2. ``docs`` number of documents TO BE GENERATED by lda, giving this parameter lda will be directly set to generative mode
3.  ``virality`` virality of the doc; if virality is high, exponent of the power law is high and threshold for activation is low.

##### Diffusion
4.  ``steps`` steps of the diffusion simulation
5.  ``actives`` percentage of active nodes with respect to the total number of nodes in the intial configuration (before diffusion) for each doc.

##### Node2Vec
6.	``dimensions``        Number of dimensions for node2vec. Default 128
7.	``walk-length``       length of walk per source. Default 80
8.	``num-walks``       number of walks per source. Default 10
9.	``window-size``      context size for optimization. Default 10
10.	``iter``           number of epochs in SGD
11.	``workers``     number of parallel workers. Default 8
12.	``p``                manually set BFS parameter; else: it is set by H
13.	``q``                 manually set DFS parameter; else: it is set by H

##### Input and Output
14.	``graph``       Input path of the graph edgelist
15.	``weighted``            boolean specifying (un)weighted. Default unweighted
16.	``unweighted``
17.	``directed``            graph is (un)directed. Default undirected
18.	``undirected``
19.	``docs-folder``  Input path of the documents folder
20.	``output``       Outputs path
21.	``format``       Outputs format
  
22.	``seed``         Seed (int) for random distribution extraction


### Citing


	@inproceedings{,
	author = {},
	 title = {},
	 booktitle = {Proceedings},
	 year = {2019}
	}


### Miscellaneous

Please feel free .. 

*Note:* This is only a reference implementation analysis and more details are provided by the thesis.
