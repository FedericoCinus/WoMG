#dealing with path (WoMG is not a library for now)

# import os
# import pathlib
# if not str(pathlib.Path.cwd()).endswith('src'):
#   src_path = pathlib.Path.cwd() / "src" / "womg"
#   os.sys.path.insert(0, str(src_path))
# if str(pathlib.Path.cwd()).endswith('examples'):
#   src_path = pathlib.Path.cwd().parent / "src" / "womg"
#   os.sys.path.insert(0, str(src_path))
#print(pathlib.Path.cwd())

##################################################

import click
from womg.network.tn import TN
from womg.topic.lda import LDA
from womg.diffusion.tlt import TLT
from womg.utils.distributions import set_seed
from womg.utils.saver import TxtSaver



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

    saver.save_mapping(network_model)
    saver.save_items_descript(topic_model)
    saver.save_topics_descript(topic_model)



def womg_main(graph=None,
              docs_path=None,
              items_descr_path=None,
              numb_topics=15,
              numb_docs=None,
              numb_steps=100,
              homophily=.5,
              gn_strength=0,
              infl_strength=None,
              virality_exp=1.5,
              virality_resistance=1.,
              interests_path=None,
              int_mode='rand',
              weighted=False, directed=False,
              path_out=None,
              seed=None,
              walk_length=100,
              num_walks=10,  window_size=10,
              iiter=1, workers=1,
              p=1, q=1,
              beta=0.01,
              norm_prior=False,
              alpha_value=2.,
              beta_value=2.,
              prop_steps=1000,
              progress_bar=False,
              save_all=False,
              save_int=False,
              save_infl=False,
              save_keyw=False,
              single_activator=False):
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
        Default 0.5

    gn_strength : float
        god node influence stength. Default 1

    infl_strength : float
        [0, 1]
        relative strength of influence with respect to the interests avg norm.
        If it is 1 then the influence vectors have the same norm as the interests.
        If 0 the influence vectors take no part in the propagation formula.
        Default: infl_strength = 1-H

    virality : float
        exponent of the powerlaw distribution for documents viralities.
        P(x; a) = x^{-a}, 0 <= x <=1. Deafault a=1


    int_mode : str
        defines the method for generating nodes' interests.
        4 choices: 'n2i', 'rand', 'prop_int', 'nmf'
        Default setting is rand


    graph : str or nx obj
        input path of the graph edgelist or nx object

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


    beta : float
        beta cost parameter for Beta-VAE loss term. Default  0.01

    norm_prior : bool
        choose half normal distribution as prior function for Beta-VAE loss term. Default False -> beta distribution

    alpha_value : float
        alpha value for the alpha vec of the Beta prior distribution. Default 2.
    beta_value : float
        beta value for the beta vec of the Beta prior distribution. Default 2.


    prop_steps : int
        [propagation of interests param] sets the number of steps in the propagation
        high value imposes high homophily in the interests

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
        network_model = TN( graph=graph,
                            numb_topics=numb_topics, homophily=homophily,
                            weighted=weighted, directed=directed,

                            interests_path=interests_path,
                            gn_strength=gn_strength,
                            infl_strength=infl_strength,
                            p=p, q=q,
                            num_walks=num_walks, walk_length=walk_length,
                            window_size=window_size,
                            workers=workers, iiter=iiter,
                            progress_bar=progress_bar,
                            beta=beta,
                            norm_prior=norm_prior,
                            alpha_value=alpha_value,
                            beta_value=beta_value,
                            prop_steps=prop_steps,
                            seed=seed)
        network_model.network_setup(int_mode=int_mode)

        topic_model = LDA(numb_topics=numb_topics,
                          numb_docs=numb_docs,
                          docs_path=docs_path,
                          items_descr_path=items_descr_path)
        topic_model.fit()
        print(virality_exp)
        topic_model.set_docs_viralities(virality_exp=virality_exp)

        diffusion_model = TLT(network_model=network_model,
                              topic_model=topic_model,
                              path_out=path_out,
                              progress_bar=progress_bar,
                              single_activator=single_activator,
                              virality_resistance=virality_resistance)
        diffusion_model.diffusion_setup()
        diffusion_model.run(numb_steps=numb_steps)
        #diffusion_model.save_threshold_values(path_out)

    finally:
        save(network_model=network_model,
             topic_model=topic_model,
             diffusion_model=diffusion_model,
             path=path_out,
             save_all=save_all,
             save_int=save_int,
             save_infl=save_infl,
             save_keyw=save_keyw)


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
                    help='0<=H<=1 :degree of homophily decoded from the given network. Default 0.5',
                    type=click.FloatRange(0, 1, clamp=True))
@click.option('--gn_strength', default=0,
                    help='Influence strength of the god node for initial configuration. Default 0',
                    type=float)
@click.option('--infl_strength', type=float, default=None,
                    help='Percentage of strength of the influence vecs with respect to interests vecs. Default 1')
@click.option('--virality_exp', metavar='V', default=1.5,
                    help='Exponent of the pareto distribution for documents viralities.',
                    type=float)
@click.option('--virality_resistance', metavar='V', default=1.,
                    help='Virality resistance factor r',
                    type=float)

@click.option('--graph', default=None,
                    help='Input path of the graph edgelist or nx object', type=str)
@click.option('--interests_path', default=None,
                    help='Input path of the ginterests table', type=str)

@click.option('--int_mode', type=str,
                    help="defines the method for generating nodes' interests. 3 choices: 'n2i', 'rand', 'prop_int'. Default 'rand' ",
                    default='rand')

@click.option('--weighted', is_flag=True,
                    help='boolean specifying (un)weighted. Default  unweighted', default=False)

@click.option('--directed', is_flag=True,
                    help='graph is (un)directed. Default  undirected',
                    default=False)


@click.option('--docs_folder', metavar='DOCS', default=None,
                    help='Input  path of the documents folder', type=str)
@click.option('--items_descr_path', default=None,
                    help='Input  path items description file representing each item in the topics space. Format: topic_index [topic-dim vec]', type=str)
@click.option('--output', default=None, help='Outputs path')
@click.option('--seed', help='Seed (int) for random distribution extractions',
                    type=int, required=False)



@click.option('--walk_length', metavar='w', type=int, default=80,
                    help='length of walk per source. Default 80')

@click.option('--num_walks', metavar='nw', type=int, default=10,
                    help='number of walks per source. Default 10')

@click.option('--window_size', metavar='ws', type=int, default=10,
                    help='context size for optimization. Default 10')

@click.option('--iiter', default=1, type=int,
                  help='number of epochs in SGD')

@click.option('--workers', type=int, default=8,
                    help='number of parallel workers. Default 8')

@click.option('--p', type=float, default=1,
                    help='manually set BFS parameter; else: it is set by H')

@click.option('--q', type=float, default=1,
                    help='manually set DFS parameter; else: it is set by H')
@click.option('--progress_bar', is_flag=True,
                    help='boolean for specifying the progress bar related to the environment if True progress_bar=tqdm_notebook -> Jupyter progress_bar; if False progress_bar=tqdm. Default False ',
                    default=False)
@click.option('--beta', type=float, default=0.01,
                    help='beta cost parameter for Beta-VAE loss term. Default  0.01')
@click.option('--norm_prior', is_flag=True, default=False,
                    help='choose half normal distribution as prior function for Beta-VAE loss term. Default  beta distribution')
@click.option('--alpha_value', type=float, default=2.,
                    help='alpha value for the alpha vec of the Beta prior distribution. Default 2.')
@click.option('--beta_value', type=float, default=2.,
                    help='beta value for the beta vec of the Beta prior distribution. Default 2.')

@click.option('--prop_steps', type=int, default=5000,
                    help='propagation steps for interests generation in propagation mode (--int_mode prop_int). Default 5000')

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
@click.option('--single_activator', is_flag=True,
                    help='if True we have at most one activator per item, else god node will activate all nodes beyond threshold',
                    default=False)

def main_cli(graph,
             items_descr_path,
             topics, docs, steps, homophily,
             virality_exp,
             virality_resistance,
             interests_path,
             int_mode,
             weighted, directed, docs_folder,
             gn_strength,
             infl_strength,
             output, seed,
             walk_length,
             num_walks, window_size,
             iiter, workers,
             p, q,
             beta,
             norm_prior,
             alpha_value,
             beta_value,
             prop_steps,
             progress_bar,
             single_activator,
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
    womg_main(graph=graph,
              docs_path=docs_folder,
              items_descr_path=items_descr_path,
              numb_topics=topics, numb_docs=docs,
              numb_steps=steps, homophily=homophily,
              gn_strength=gn_strength,
              infl_strength=infl_strength,
              virality_exp=virality_exp,
              virality_resistance=virality_resistance,
              interests_path=interests_path,
              int_mode=int_mode,
              weighted=weighted, directed=directed,
              path_out=output,
              seed=seed,
              walk_length=walk_length,
              num_walks=num_walks, window_size=window_size,
              iiter=iiter, workers=workers,
              p=p, q=q,
              beta=beta,
              norm_prior=norm_prior,
              alpha_value=alpha_value,
              beta_value=beta_value,
              prop_steps=prop_steps,
              progress_bar=progress_bar,
              save_all=save_all,
              save_int=save_int,
              save_infl=save_infl,
              save_keyw=save_keyw,
              single_activator=single_activator)


if __name__ == '__main__':
    main_cli()
