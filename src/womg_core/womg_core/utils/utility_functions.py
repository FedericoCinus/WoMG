'''Utility functions
'''
import pathlib
import pickle
import networkx as nx


def read_edgelist(path, weighted, directed):
    '''Reads the input network in networkx.
    '''
    if weighted:
        graph = nx.read_edgelist(path, nodetype=int,
                                 data=(('weight', float),),
                                 create_using=nx.DiGraph())
    else:
        graph = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph())
        for edge in graph.edges():
            graph[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        graph = graph.to_directed()

    # mapping labels
    mapping = {}
    identity_map = 0
    for new_label, old_label in enumerate(sorted(graph.nodes())):
        if new_label == old_label:
            identity_map += 1
        mapping[old_label] = new_label
    if identity_map == graph.number_of_nodes():
        return graph, None
    return nx.relabel_nodes(graph, mapping), mapping


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
    with open(file_path, 'rb') as file:
        rfile = pickle.load(file)
    return rfile

def read_docs(path, verbose=False):
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
    onlyfiles = [f for f in onlyfiles if not f.startswith('.')]
    docs = []
    if verbose:
        print(onlyfiles)
    for file in onlyfiles:
        f_path = pathlib.Path(path) / str(file)
        with open(f_path, 'rb') as file:
            doc_list = [j for j in file]
            if verbose:
                print(doc_list)
            docs.append(doc_list)
    return docs


def find_numb_nodes(graph):
    '''
    Finds out number of nodes for a formatted graph
    '''
    maxx = 0
    for key in graph.keys():
        for i in range(2):
            if key[i] > maxx:
                maxx = key[i]
    return maxx
