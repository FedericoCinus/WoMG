# WoMG: Word of Mouth Generator

This repository provides a reference implementation of *WoMG* as described in the thesis:<br>
> WoMG: A synthetic Word-of-Mouth cascades Generator.<br>
> Federico Cinus.<br>
> M.Sc. Physics of Complex Systems, Università degli studi di Torino, Turin, 2017-2018.<br>
> <Insert paper link>

The *WoMG* software generates synthetic datasets of documentss propagation on network. 
It starts with any (un)directed, (un)weighted graph and a collection of documents and outputs the propagation DAGs of the docs through the network. 
Diffusion process is guided by the nodes underlying preferences. Please check the [project page]() for more details. 

### Basic Usage

#### Example
To run *WoMG* on Les Miserables network, execute the following command from the project home directory:<br/>

	``python main.py``

notice that it will generates 100 documents to be spread over the default network (Les Miserables).

#### Options
You can check out the other options available to use with *WoMG* using:<br/>

	``python main.py --help``

#### Input
1. [Network] The supported input format is an edgelist (txt extension):
	
	node1_id_int node2_id_int <weight_float, optional>
		
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags. You have to specify the edgelist path using the *graph* argument:

	``python main.py --graph /this/is/an/example/path/Graph_Folder/edgelist.txt``

2. [Documents] The supported input format for documents collection (corpus) is txt. You have to specify the folder path containing them using the *docs_folder* argument:

  	``python main.py --docs_folder /this/is/an/example/path/Corpus_Folder``

  
#### Output
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

  ``python main.py --format pickle``
  ``python main.py --format txt``
  
and specify the output folder path:

  ``python main.py --output /this/is/an/example/path/Output_Folder``




### Usage for researchers


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
