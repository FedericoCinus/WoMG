'''
Reference implementation of node2vec.

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec
Knowledge Discovery and Data Mining (KDD), 2016
'''
import sys
import click
import pathlib
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.decomposition import NMF
from n2i.node2vec import node2vec, read_graph, save_emb

curr_path = str(pathlib.Path.cwd())

def n2i_nx_graph(nx_graph,
         topics=15,
         weighted=False, directed=False,
         fast=False, seed=None,
         dimensions=128, walk_length=80,
         num_walks=10, window_size=10,
         iiter=1, workers=8,
         p=1, q=1,
         normalize=False,
         translate=True,
         reduce=True,
         verbose=False,
         use_tf=False,
         beta=0,
         prior='half_norm'):

    # seed
    if seed != None:
        random.seed(seed)
        np.random.seed(seed)

    numb_nodes = nx.number_of_nodes(nx_graph)
    model = node2vec(weighted=weighted,
                    graph=nx_graph,
                    directed=directed,
                    p=p, q=q,
                    num_walks=num_walks,
                    walk_length=walk_length,
                    dimensions=dimensions,
                    window_size=window_size,
                    workers=workers,
                    iiter=iiter,
                    verbose=verbose,
                    use_tf=use_tf,
                    beta=beta,
                    prior=prior)
    if reduce:
        # Translation
        if not use_tf:
            # translation constant
            minim = np.amin(model)
            print('Translating')
            model = model + abs(minim)
        # NMF Reduction
        print('Reducing')
        if verbose:
            print('Reducing dimensions from ', dimensions,' to ', topics)
        nmf = NMF(n_components=topics, random_state=42, max_iter=1000)
        _right = nmf.fit(model).components_
        left = nmf.transform(model)
        embeddings = left

    # NO REDUCTION
    else:
        if translate and not use_tf:
            # translation constant
            minim = np.amin(model)
            print('Translating')
            model = model + abs(minim)
        embeddings = model


    return embeddings

def n2i_main(topics=15,
         graph=None, fast=False,
         weighted=False, directed=False,
         output=curr_path, seed=None,
         dimensions=128, walk_length=80,
         num_walks=10, window_size=10,
         iiter=1, workers=8,
         p=1, q=1,
         beta=0,
         prior='half_norm',
         normalize=False,
         translate=True,
         reduce=True,
         verbose=False):
    '''



    Creates interests vector for each node using node2vec algorithm and
    saves interests in a file.

    3 steps:
        1. finding node2vec embeddings
        2. translation to positive axes of the embeddings
        3. reduction with NMF to _topics dimensions

    Parameters
    ----------
    - topics : int
        Number of topics in the topic model. Default 15. K<d
    - graph : str
        Input path of the graph edgelist
    - fast : bool
        defines the method for generating nodes' interests. Two choices: 'node2interests', 'random'. Default setting is False -> 'node2interests
    - weighted : bool
        boolean specifying (un)weighted. Default  unweighted
    - directed : bool
        graph is (un)directed. Default  undirected
    - output : str
        Outputs path
    - seed : int
        Seed (int) for random distribution extractions
    - dimensions : int
        Number of dimensions for node2vec. Default 128
    - walk_length : int
        ength of walk per source. Default 80
    - num_walks : int
        number of walks per source. Default 10
    - window_size : int
        context size for optimization. Default 10
    - iiter : int
        number of epochs in SGD
    - workers : int
        number of parallel workers. Default   8
    - p : float
        manually set BFS parameter
    - q : float
        manually set DFS parameter
    - norm : bool
      choose if interests have to be normalize (True) or not (False)
    - transl : bool
      choose if interests have to be translated to positive axes
    - reduce : bool
        if True reduces dimensions with NMF



    Notes
    -----
    - Arrays are translated in order to have non-negative entries
    - Dimension is reduced to the number of topics using NMF, which mantains
      positivity
    '''
    if topics >= dimensions:
        print('Topics have to be less than dimensions')
        sys.exit()

    nx_graph = read_graph(weighted=weighted, graph=graph, directed=directed)

    emb = n2i_nx_graph(
         nx_graph=nx_graph,
         topics=topics,
         weighted=weighted, directed=directed,
         fast=fast,
         seed=seed,
         dimensions=dimensions,
         walk_length=walk_length,
         num_walks=num_walks,
         window_size=window_size,
         iiter=iiter,
         workers=workers,
         p=p, q=q,
         beta=beta,
         prior=prior,
         normalize=normalize,
         translate=translate,
         reduce=reduce,
         verbose=verbose)

    save_emb(emb=emb, path=output, verbose=verbose)


@click.command()
@click.option('--topics', metavar='K', default=15,
                    help='Number of topics in the topic model. Default 15. K<d ',
                    type=int)
@click.option('--graph', default=None,
                    help='Input path of the graph edgelist', type=str)

@click.option('--fast', is_flag=True,
                    help="defines the method for generating nodes' interests. Two choices: 'node2interests', 'random'. Default setting is False -> 'node2interests",
                    default=False)

@click.option('--weighted', is_flag=True,
                    help='boolean specifying (un)weighted. Default  unweighted', default=False)

@click.option('--directed', is_flag=True,
                    help='graph is (un)directed. Default  undirected',
                    default=False)
@click.option('--output', default=curr_path, help='Outputs path')
@click.option('--seed', help='Seed (int) for random distribution extractions',
                    type=int, required=False)


@click.option('--dimensions', metavar='d', type=int, default=128,
                    help='Number of dimensions for node2vec. Default 128')

@click.option('--walk_length', metavar='w', type=int, default=80,
                    help='length of walk per source. Default 80')

@click.option('--num_walks', metavar='nw', type=int, default=10,
                    help='number of walks per source. Default 10')

@click.option('--window_size', metavar='ws', type=int, default=10,
                    help='context size for optimization. Default 10')

@click.option('--iiter', default=1, type=int,
                  help='number of epochs in SGD. Default 1')

@click.option('--workers', type=int, default=8,
                    help='number of parallel workers. Default   8')

@click.option('--p', type=float, default=1,
                    help='manually set BFS parameter')

@click.option('--q', type=float, default=1,
                    help='manually set DFS parameter')
@click.option('--normalize', is_flag=True,
                    help='normalize embeddings',
                    default=False)
@click.option('--translate', is_flag=True,
                    help='translate embeddings to potsitive axes',
                    default=True)
@click.option('--beta', type=int, default=0,
                    help='beta parameter for Beta-VAE loss term. Default  0')
@click.option('--prior', type=str, default='half_norm',
                    help='prior function for Beta-VAE loss term. Default  half_norm')
@click.option('--reduce', is_flag=True,
                    help='reduce dimension with NMF',
                    default=True)
@click.option('--verbose', is_flag=True,
                    help='n2i rpivdes all prints',
                    default=False)
def main_cli(topics,
         graph, fast,
         weighted, directed,
         output, seed,
         dimensions, walk_length,
         num_walks, window_size,
         iiter, workers,
         p, q,
         beta,
         prior,
         normalize,
         translate,
         reduce,
         verbose):
    '''


    Creates interests vector for each node using node2vec algorithm and
    saves interests in a file.

    3 steps:
        1. finding node2vec embeddings
        2. translation to positive axes of the embeddings
        3. reduction with NMF to _topics dimensions
    '''
    n2i_main(topics=topics,
             graph=graph, fast=fast,
             weighted=weighted, directed=directed,
             output=output, seed=seed,
             dimensions=dimensions, walk_length=walk_length,
             num_walks=num_walks, window_size=window_size,
             iiter=iiter, workers=workers,
             p=p, q=q,
             beta=beta,
             prior=prior,
             normalize=normalize,
             translate=translate,
             reduce=reduce,
             verbose=verbose)

if __name__ == '__main__':
    main_cli()
