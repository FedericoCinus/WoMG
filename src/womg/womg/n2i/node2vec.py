'''
Reference implementation of node2vec.

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''

from n2i.tfoptimizer import Word2vec as TfWord2vec

import pathlib
import networkx as nx
from n2i.graph import Graph
import numpy as np
from gensim.models import Word2Vec as GensimWord2Vec

def read_graph(weighted, graph, directed):
    '''
    Reads the input network in networkx.
    '''
    #print(graph)
    if weighted:
        G = nx.read_edgelist(graph, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(graph, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    return G

def learn_embeddings(number_of_nodes, walks, dimensions, window_size, workers, iiter, use_tf=True,
    beta=0, prior='half_norm',
    verbose=True):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    if use_tf:
        model = TfWord2vec(number_of_nodes, embedding_size=dimensions, beta=beta, prior=prior)
        return model.run(walks, iiter=iiter, window=window_size, verbose=verbose)
    else:
        walks = [list(map(str, walk)) for walk in walks]
        model = GensimWord2Vec(walks, size=dimensions, window=window_size, min_count=0,
                         sg=1, workers=workers, iter=iiter)

        embeddings = []
        for node in range(number_of_nodes):
            curr_emb = np.array([model.wv[str(node)][_] for _ in range(dimensions)])
            embeddings.append(curr_emb)
        return np.array(embeddings)

def node2vec(weighted, graph, directed, p, q, num_walks, walk_length,
             dimensions, window_size, workers, iiter, verbose, use_tf=True,
             beta=0, prior='half_norm', alpha_value=2., beta_value=2.):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    if type(graph) == str:
        nx_G = read_graph(weighted, graph, directed)
    else:
        nx_G = graph

    G = Graph(nx_G, directed, p, q, verbose=verbose)
    G.preprocess_transition_probs()

    walks = G.simulate_walks(num_walks, walk_length)
    '''
    g2v = Node2Vec()
    walks = g2v.simulate_walks(nx_G, walklen=walk_length,
                                epochs=num_walks,
                                return_weight=1/p,
                                neighbor_weight=1/q,
                                threads=workers)
    #print(walks)
    '''
    emb_model = learn_embeddings(nx_G.number_of_nodes(), walks, dimensions, window_size,
                                 workers, iiter, verbose=verbose, use_tf=use_tf, beta=beta,
                                 prior=prior)

    return emb_model

def save_emb(emb, path, verbose=False):
    '''
    saves embeddings in path with the following format :
    node_id [embeddings]
    '''
    file_name = make_filename('emb', pathlib.Path(path))
    with open(file_name, 'w') as f:
        for node in emb.keys():
            tmp_emb = [_ for _ in emb[node]]
            f.write(str(node) + ' ' + str(tmp_emb) + '\n')
    if verbose:
        print('Embeddings have been saved in ', path)

def make_filename(name, output_dir, new_file=True):
    '''
    Returns the correct filename checking the file "name" is in "output_dir"
    If the file already exists it returns a filename with an incrementing
    index.

    Parameters
    ----------
    - name : str
        name of the file
    - output_dir : str
        path of the "name" file

    Returns
    -------
    Filename (str)
    '''
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = output_dir / str(name + "0.txt")
    sim_numb = 0

    while pathlib.Path(filename).exists():
        sim_numb+=1
        filename = output_dir / str(name + str(sim_numb) + ".txt")
    if new_file:
        return filename
    else:
        sim_numb-=1
        filename = output_dir / str(name + str(sim_numb) + ".txt")
        return filename
