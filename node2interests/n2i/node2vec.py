'''
Reference implementation of node2vec.

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''
import pathlib
import networkx as nx
from n2i.graph import Graph
from gensim.models import Word2Vec

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
    return G

def learn_embeddings(walks, dimensions, window_size, workers, iiter):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0,
                     sg=1, workers=workers, iter=iiter)
    #model.wv.save_word2vec_format(args.output)

    return model.wv

def node2vec(weighted, graph, directed, p, q, num_walks, walk_length,
                  dimensions, window_size, workers, iiter, verbose):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    if type(graph) == str:
        nx_G = read_graph(weighted, graph, directed)
    else:
        nx_G = graph
    G = Graph(nx_G, directed, weighted, p, q, verbose=verbose)
    G.format_graph()
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    emb_model = learn_embeddings(walks, dimensions, window_size,
                                 workers, iiter)

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
