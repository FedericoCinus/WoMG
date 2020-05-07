'''
WoMG main function
'''
import click
from womg_core.propagation import Propagation
from womg_core.network.tn import TN
from womg_core.topic.lda import LDA
from womg_core.diffusion.tlt import TLT
from womg_core.utils.distributions import set_seed
from womg_core.utils.saver import TxtSaver




def save(network_model, topic_model, diffusion_model,
         path,
         save_int, save_infl, save_keyw):
    '''Saves the network, items descriptions, topics descriptions
       in file.
    '''
    saver = TxtSaver(path)

    if save_int:
        saver.save_users_interests(network_model)
    if save_infl:
        saver.save_users_influence(network_model)
    if save_keyw:
        saver.save_items_keyw(topic_model)

    saver.save_mapping(network_model)
    saver.save_items_descript(topic_model)
    saver.save_topics_descript(topic_model)
    saver.save_propagations(diffusion_model)


def womg_main(graph=None, #pylint: disable=too-many-arguments, too-many-locals
              items_descr=None,
              numb_topics=15,
              numb_docs=None,
              numb_steps=6,
              homophily=.5,
              gn_strength=13.,
              infl_strength=12.,
              virality_exp=8.,
              virality_resistance=13.,
              interests=None,
              int_mode='nmf',
              weighted=False, directed=False,
              seed=None,
              progress_bar=True,
              single_activator=False,
              path_out=None):
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
        number of topics for documents' topic model and nodes' interests. Default 15.

    numb_docs : int
        number of docs to be generated. Default 100

    numb_steps : int
        number of time steps for diffusion

    homophily : float
        0<=H<=1 :degree of homophily decoded from the given network.
        Default 0.5

    gn_strength : float
        god node influence stength at intial time step. Default 13

    infl_strength : float
        relative strength of influence with respect to the interests
        Default: 12.

    virality : float
        exponent of the powerlaw distribution for documents viralities. Default=8


    int_mode : str
        defines the method for generating nodes' interests.
        2 choices:  'rand', 'nmf'
        Default setting is 'nmf'


    graph : str or nx obj
        input path of the graph edgelist or nx object. If nx obj, womg creates a
        copy of the input graph.

    weighted : bool
        boolean specifying (un)weighted. Default  unweighted

    directed : bool
        graph is (un)directed. Default  undirected


    seed : int
        seed (int) for random distribution extractions

    progress_bar : bool
        boolean for specifying the progress bar related to the environment
        if True progress_bar=tqdm_notebook -> Jupyter progress_bar;
        if False progress_bar=tqdm. Default False

    '''
    set_seed(seed)
    network_model = TN(graph=graph,
                       numb_topics=numb_topics, homophily=homophily,
                       weighted=weighted, directed=directed,
                       interests=interests,
                       gn_strength=gn_strength,
                       infl_strength=infl_strength,
                       progress_bar=progress_bar,
                       seed=seed)
    network_model.network_setup(int_mode=int_mode)

    topic_model = LDA(numb_topics=numb_topics,
                      numb_docs=numb_docs,
                      items_descr=items_descr,
                      progress_bar=progress_bar)
    topic_model.fit()
    #print(virality_exp)
    topic_model.set_docs_viralities(virality_exp=virality_exp)

    diffusion_model = TLT(network_model=network_model,
                          topic_model=topic_model,
                          path_out=path_out,
                          progress_bar=progress_bar,
                          single_activator=single_activator,
                          virality_resistance=virality_resistance)
    diffusion_model.diffusion_setup()
    diffusion_model.run(numb_steps=numb_steps)
    propagation = Propagation(network_model,
                              topic_model,
                              diffusion_model)


    return propagation

@click.command()
@click.option('--topics', metavar='K', default=15,
              help='Number of topics in the topic model. Default 15. K<d ',
              type=int)
@click.option('--docs', metavar='D', default=None,
              help='Number of docs to be generated. Default 100',
              type=int)
@click.option('--steps', metavar='T', default=6,
              help='Number of time steps for diffusion',
              type=int)
@click.option('--homophily', metavar='H', default=0.5,
              help='0<=H<=1 :degree of homophily decoded from the given network. Default 0.5',
              type=click.FloatRange(0, 1, clamp=True))
@click.option('--gn_strength', default=13.,
              help='Influence strength of the god node for initial configuration. Default 13.',
              type=float)
@click.option('--infl_strength', type=float, default=None,
              help='Influence strength of nodes with respect to interests vecs. Default 12.')
@click.option('--virality_exp', metavar='V', default=8.,
              help='Exponent of the pareto distribution for documents viralities. Default 8.',
              type=float)
@click.option('--virality_resistance', metavar='V', default=13.,
              help='Virality resistance factor r. Default 13.',
              type=float)

@click.option('--graph', default=None,
              help='Input path of the graph edgelist or nx object', type=str)
@click.option('--interests', default=None,
              help='Input path of the ginterests table', type=str)

@click.option('--int_mode', type=str,
              help="defines the method for generating nodes' interests. 2 choices: 'rand', 'nmf'. Default 'nmf' ",
              default='nmf')

@click.option('--weighted', is_flag=True,
              help='boolean specifying (un)weighted. Default  unweighted', default=False)

@click.option('--directed', is_flag=True,
              help='graph is (un)directed. Default  undirected',
              default=False)

@click.option('--items_descr', default=None,
              help='Input path or dictionary of items description file representing each item in the topics space. Format: topic_index [topic-dim vec]', type=str)
@click.option('--output', default=None, help='Outputs path')
@click.option('--seed', help='Seed (int) for random distribution extractions',
              type=int, required=False)


@click.option('--progress_bar', is_flag=True,
              help='boolean for specifying the progress bar related to the environment if True progress_bar=tqdm_notebook -> Jupyter progress_bar; if False progress_bar=tqdm. Default False ',
              default=True)

@click.option('--save_int', is_flag=True,
              help='if True WoMG saves the interests vector for each node',
              default=False)
@click.option('--save_infl', is_flag=True,
              help='if True WoMG saves the influence vector for each node', #pylint: disable=too-many-arguments, too-many-locals
              default=False)
@click.option('--save_keyw', is_flag=True,
              help='if True WoMG saves the keywords in a bow format for each document',
              default=False)
@click.option('--single_activator', is_flag=True,
              help='if True we have at most one activator per item, else god node will activate all nodes beyond threshold',
              default=False)

def main_cli(graph,
             items_descr,
             topics, docs, steps, homophily,
             virality_exp,
             virality_resistance,
             interests,
             int_mode,
             weighted, directed,
             gn_strength,
             infl_strength,
             output, seed,
             progress_bar,
             single_activator,
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
    prop = womg_main(graph=graph,
                     items_descr=items_descr,
                     numb_topics=topics, numb_docs=docs,
                     numb_steps=steps, homophily=homophily,
                     gn_strength=gn_strength,
                     infl_strength=infl_strength,
                     virality_exp=virality_exp,
                     virality_resistance=virality_resistance,
                     interests=interests,
                     int_mode=int_mode,
                     weighted=weighted, directed=directed,
                     seed=seed,
                     progress_bar=progress_bar,
                     single_activator=single_activator,
                     path_out=output)

    save(network_model=prop.network_model,
         topic_model=prop.topic_model,
         diffusion_model=prop.diffusion_model,
         path=output,
         save_int=save_int,
         save_infl=save_infl,
         save_keyw=save_keyw)


if __name__ == '__main__':
    main_cli() # pylint: disable=no-value-for-parameter
