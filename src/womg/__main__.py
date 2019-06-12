#dealing with path (WoMG is not a library for now)

import os
import pathlib
if not str(pathlib.Path.cwd()).endswith('src'):
  src_path = pathlib.Path.cwd() / "src" / "womg"
  os.sys.path.insert(0, str(src_path))
if str(pathlib.Path.cwd()).endswith('examples'):
  src_path = pathlib.Path.cwd().parent / "src" / "womg"
  os.sys.path.insert(0, str(src_path))
#print(pathlib.Path.cwd())

##################################################

import click
from network.tn import TN
from topic.lda import LDA
from diffusion.tlt import TLT
from utils.utility_functions import cleaning
from utils.distributions import set_seed
from utils.saver import TxtSaver



def save(network_model, topic_model, diffusion_model,
         path, save_all,
         save_int, save_infl, save_keyw):
    saver = TxtSaver(path)
    if save_all == True:
        save_int = save_infl = save_keyw = True

    if save_int:
        saver.save_users_interests(network_model)
    if save_infl:
        saver.save_users_influence(network_model)
    if save_keyw:
        saver.save_items_keyw(topic_model)

    saver.save_items_descript(topic_model)
    saver.save_topics_descript(topic_model)
    #saver.save_propagations(diffusion_model)



def womg_main(numb_topics, numb_docs,
              numb_steps, homophily,
              actives_perc, virality,
              path_in_graph,
              fast,
              weighted, directed,
              god_node, docs_path,
              path_out,
              seed,
              dimensions, walk_length,
              num_walks,  window_size,
              iiter, workers,
              p, q,
              progress_bar,
              save_all,
              save_int,
              save_infl,
              save_keyw):
    '''


    --------------------------------------------------------------------
    WoMG main function:

    The *WoMG* software generates synthetic datasets of documents cascades on network.
    It starts with any (un)directed, (un)weighted graph and a collection of
    documents and it outputs the propagation DAGs of the docs through the network.
    Diffusion process is guided by the nodes underlying preferences.
    Please check the github page for more details.


    Parameters
    ----------
    numb_topics : int
        number of topics in the topic model. Default 15. K<d

    numb_docs : int
        number of docs to be generated. Default 100

    numb_steps : int
        number of time steps for diffusion

    homophily : float
        0<=H<=1 :degree of homophily decoded from the given network.
        1-H is degree of influence between nodes;
        reccommended values are: 0, 0.5, 1. Default 0.5

    actives_perc : float
        maximum percentage of active nodes in first step of simulation on an item
        with respect to the number of nodes. Default 0.05

    virality : float
        exponent of the powerlaw distribution for documents viralities.
        P(x; a) = x^{-a}, 0 <= x <=1. Deafault a=1


    fast : bool
        defines the method for generating nodes' interests.
        Two choices: 'node2interests', 'random'.
        Default setting is True -> 'random'


    path_in_graph : str
        input path of the graph edgelist

    weighted : bool
        boolean specifying (un)weighted. Default  unweighted

    directed : bool
        graph is (un)directed. Default  undirected


    docs_path : str
        input  path of the documents folder

    path_out : str
        outputs path


    seed : int
        seed (int) for random distribution extractions


    dimensions : int
        [node2vec param] number of dimensions. Default 128

    walk_length : int
        [node2vec param] length of walk per source. Default 80

    num_walks : int
        [node2vec param] number of walks per source. Default 10

    window_size : int
        [node2vec param] context size for optimization. Default 10

    iiter : int
        [node2vec param] number of epochs in SGD

    workers: int
        [node2vec param] number of parallel workers. Default 8

    p : float
        [node2vec param] manually set BFS parameter; else: it is set by H

    q : float
        [node2vec param] manually set DFS parameter; else: it is set by H

    progress_bar : bool
        boolean for specifying the progress bar related to the environment
        if True progress_bar=tqdm_notebook -> Jupyter progress_bar;
        if False progress_bar=tqdm. Default False

    save_int : bool
        if True WoMG saves the interests vector for each node

    save_infl : bool
        if True WoMG saves the influence vector for each node

    save_keyw : bool
        if True WoMG saves the keywords in a bow format for each document

    save_all : bool
        save all non-optional outputs
    '''

    try:
        set_seed(seed)
        network_model = TN(numb_topics=numb_topics, homophily=homophily,
                            god_node=False,
                            weighted=weighted, directed=directed,
                            path_in_graph=path_in_graph,
                            p=p, q=q,
                            num_walks=num_walks, walk_length=walk_length,
                            dimensions=dimensions, window_size=window_size,
                            workers=workers, iiter=iiter,
                            progress_bar=progress_bar)
        network_model.network_setup(fast=fast)

        topic_model = LDA(numb_topics=numb_topics,
                          numb_docs=numb_docs,
                          path_in=docs_path)
        topic_model.fit()
        topic_model.set_docs_viralities(virality=virality)

        diffusion_model = TLT(network_model=network_model,
                              topic_model=topic_model,
                              actives_perc=actives_perc,
                              path_out=path_out,
                              progress_bar=progress_bar)
        diffusion_model.diffusion_setup()
        diffusion_model.run(numb_steps=numb_steps)

        save(network_model=network_model,
             topic_model=topic_model,
             diffusion_model=diffusion_model,
             path=path_out,
             save_all=save_all,
             save_int=save_int,
             save_infl=save_infl,
             save_keyw=save_keyw)
    finally:
        cleaning()

@click.command()
@click.option('--topics', metavar='K', default=15,
                    help='Number of topics in the topic model. Default 15. K<d ',
                    type=int)
@click.option('--docs', metavar='D', default=None,
                    help='Number of docs to be generated. Default 100',
                    type=int)
@click.option('--steps', metavar='T', default=100,
                    help='Number of time steps for diffusion',
                    type=int)
@click.option('--homophily', metavar='H', default=0.5,
                    help='0<=H<=1 :degree of homophily decoded from the given network. 1-H is degree of influence between nodes; reccommended values are: 0, 0.5, 1. Default 0.5',
                    type=float)
@click.option('--actives', metavar='A', default=0.5,
                    help='Maximum percentage of active nodes in first step of simulation on an item with respect to the number of nodes. Default 0.5',
                    type=float)
@click.option('--virality', metavar='V', default=1,
                    help='Exponent of the powerlaw distribution for documents viralities. P(x; a) = x^{-a}, 0 <= x <=1. Default a=1',
                    type=float)

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


@click.option('--docs_folder', metavar='DOCS', default=None,
                    help='Input  path of the documents folder', type=str)
@click.option('--output', default=None, help='Outputs path')
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
                  help='number of epochs in SGD')

@click.option('--workers', type=int, default=8,
                    help='number of parallel workers. Default   8')

@click.option('--p', type=float, default=1,
                    help='manually set BFS parameter; else: it is set by H')

@click.option('--q', type=float, default=1,
                    help='manually set DFS parameter; else: it is set by H')
@click.option('--progress_bar', is_flag=True,
                    help='boolean for specifying the progress bar related to the environment if True progress_bar=tqdm_notebook -> Jupyter progress_bar; if False progress_bar=tqdm. Default False ',
                    default=False)

@click.option('--save_int', is_flag=True,
                    help='if True WoMG saves the interests vector for each node',
                    default=False)
@click.option('--save_infl', is_flag=True,
                    help='if True WoMG saves the influence vector for each node',
                    default=False)
@click.option('--save_keyw', is_flag=True,
                    help='if True WoMG saves the keywords in a bow format for each document',
                    default=False)
@click.option('--save_all', is_flag=True,
                    help='if True WoMG saves all non-optional outputs',
                    default=False)

def main_cli(topics, docs, steps, homophily, actives,
             virality, graph, fast,
             weighted, directed, docs_folder,
             output, seed,
             dimensions, walk_length,
             num_walks, window_size,
             iiter, workers,
             p, q,
             progress_bar,
             save_all,
             save_int,
             save_infl,
             save_keyw):
    '''


    The *WoMG* software generates synthetic datasets of documents cascades on network.
    It starts with any (un)directed, (un)weighted graph and a collection of
    documents and it outputs the propagation DAGs of the docs through the network.
    Diffusion process is guided by the nodes underlying preferences.
    Please check the github page for more details.

    '''
    womg_main(numb_topics=topics, numb_docs=docs,
              numb_steps=steps, homophily=homophily,
              actives_perc=actives, virality=virality,
              path_in_graph=graph,
              fast=fast,
              weighted=weighted, directed=directed,
              god_node=False, docs_path=docs_folder,
              path_out=output,
              seed=seed,
              dimensions=dimensions, walk_length=walk_length,
              num_walks=num_walks, window_size=window_size,
              iiter=iiter, workers=workers,
              p=p, q=q,
              progress_bar=progress_bar,
              save_all=save_all,
              save_int=save_int,
              save_infl=save_infl,
              save_keyw=save_keyw)


if __name__ == '__main__':
    main_cli()
