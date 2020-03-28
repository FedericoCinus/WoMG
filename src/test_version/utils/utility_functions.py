# Utility functions
import pathlib
import pickle
import networkx as nx



class TopicsError(Exception):
    pass

class DocsError(Exception):
    pass

def read_edgelist(self, path, weighted, directed):
    '''
        Reference implementation of node2vec.

        Author: Aditya Grover

        For more details, refer to the paper:
        node2vec: Scalable Feature Learning for Networks
        Aditya Grover and Jure Leskovec
        Knowledge Discovery and Data Mining (KDD), 2016

    Reads the input network in networkx. [node2vec implementation]
    '''
    if weighted:
        #G = nx.read_edgelist(path, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
        G = nx.read_edgelist(path, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    # mapping labels
    mapping = {}
    identity_map = 0
    for new_label, old_label in enumerate(sorted(G.nodes())):
        if new_label == old_label:
            identity_map += 1
        mapping[old_label] = new_label
    if identity_map == G.number_of_nodes():
        return G, None
    else:
        return nx.relabel_nodes(G, mapping), mapping

'''
def def_numb_topics(numb_topics, numb_docs, docs_path):
    #
    Setting the numb_topics equal to the one given.
    In case no documents are given, lda will be set in generative mode
    and this class has to generate interets vectors of same dimension as
    the topic distributions of docs
    #
    if ((numb_docs == None) and (docs_path == None)):
        return 15
    elif numb_docs:
        return 15
    else:
        return numb_topics
'''


def cleaning():
    '''Removing temporary class files

    Notes
    -----
    Temporary files start with '__'
    '''
    net_file = pathlib.Path.cwd() / "__network_model"
    if pathlib.os.path.exists(net_file):
        pathlib.os.remove(net_file)
    top_file = pathlib.Path.cwd() / "__topic_model"
    if pathlib.os.path.exists(top_file):
        pathlib.os.remove(top_file)

def count_files(path):
    '''Returns number of files in a directory; files that start wtih '.' are not
       considered

    Parameters
    ----------
    path : string
        directory path in which we want to know the number of files

    Notes
    -----
    It is not recursive: it does not count files in inner directories
    '''
    files = next(pathlib.os.walk(str(path)))[2]
    for file in files:
        if file.startswith('.'):
            files.remove(file)
    return len(files)

def read_graph(file_path):
    '''
    Reads graph from a path
    '''
    with open(file_path, 'rb') as f:
        rfile = pickle.load(f)
    return rfile

def read_docs(path):
    '''
    Returns the list of items, each row is the doc list

    Parameters
    ----------
    path : str
        position of the folder

    Returns
    -------
    docs in a list format : each entry of the list is a file in a list of words
    '''
    onlyfiles = [f for f in pathlib.os.listdir(path) if pathlib.os.path.isfile(pathlib.os.path.join(path, f))]
    docs = []
    for file in onlyfiles:
        f_path = pathlib.Path(path) / str(file)
        with open(f_path, 'rb') as f:
            doc_list = [j for j in f]
            docs.append(doc_list)
    return docs


def find_numb_nodes(graph):
    '''
    Finds out number of nodes for a formatted graph
    '''
    maxx = 0
    for key in graph.keys():
        for i in range(2):
            if key[i]>maxx:
                maxx = key[i]
    return maxx
