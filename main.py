# Defining pipeline for the propgenn library usage
import os
import pathlib
src_path = pathlib.Path.cwd() / "src"
os.sys.path.insert(0, str(src_path))
import argparse
from network.tn import TN
from topic.lda import LDA
from diffusion.tlt import TLT
from utilities.utility_functions import cleaning
from utilities.distributions import set_seed


def parse_args():
    '''Parsing arguments
    '''
    default_graph = pathlib.Path.cwd() / "Input" / "Graph" / "lesmiserables_edgelist.txt"
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('--topics', metavar='K', nargs='?', default=15, help='number of topics in the topic model. Default: 15. NB It has to be less then dimensions',
                        type=int)
    parser.add_argument('--docs', metavar='D', nargs='?', default=None,  help='number of docs to be generated',
                        type=int)
    parser.add_argument('--steps', metavar='T', nargs='?', default=100,  help='number of time steps for TLT model',
                        type=int)
    parser.add_argument('--homophily', metavar='H', nargs='?', default=0.5, help='0<=H<=1 :degree of homophily decoded from the given network. 1-H is degree of influence between nodes. Reccommended values are: 0, 0.5, 1 ',
                        type=float)
    parser.add_argument('--actives', metavar='A', nargs='?', default=0.5, help='maximum percentage of active nodes in first step of simulation on an item. Percentage is calculated on the number of nodes. Default is 0.5',
                        type=float)
    parser.add_argument('--virality', metavar='V', nargs='?', default=1, help='exponent of the powerlaw distribution for documents viralities. P(x; a) = x^{-a}, 0 <= x <=1. Deafault a=1',
                        type=float)

    parser.add_argument('--graph', nargs='?', default=str(default_graph), help='Input path of the graph edgelist', type=str)

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')


    parser.add_argument('--docs_folder' , default=None, help='Input  path of the documents folder')
    parser.add_argument('--output', default=None, help='Outputs path')
    parser.add_argument('--format',  default='txt', help='Outputs format')
    parser.add_argument('--seed', dest='seed', nargs='?', help='int seed for random distribution extrations',
                        type=int, required=False)
    parser.set_defaults(directed=False)



    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='It is set by H')

    parser.add_argument('--q', type=float, default=1,
                        help='It is set by H')

    return parser.parse_args()


def graph_model(numb_topics, homophily, weighted, directed, path_in, path_out, fformat, numb_docs, docs_path, args):
    '''Reading graph with networkx
    '''
    print('Loading graph')
    network_model = TN(numb_topics=numb_topics, homophily=homophily, god_node=False, weighted=weighted, directed=directed, path_in=path_in, numb_docs=numb_docs, docs_path=docs_path, args=args)
    network_model.save_model_attr(path=path_out, fformat=fformat)

def topic_model(numb_topics, numb_docs, virality, path_in, path_out, fformat):
    '''Generates a topic model
    '''
    topic_model = LDA(numb_topics=numb_topics, numb_docs=numb_docs, virality=virality, path_in=path_in)
    topic_model.save_model_attr(path=path_out, fformat=fformat)

def diffusion_model(numb_steps, actives_perc, path_out, fformat):
    '''Generates diffusion propagations
    '''
    diffusion_model = TLT(numb_steps=numb_steps, actives_perc=actives_perc, path_out=path_out, fformat=fformat, out_format='dict')

def main(args):
    '''Pipeline for the propagation generation
    '''
    try:
        set_seed(args.seed)
        graph_model(numb_topics=args.topics, homophily=args.homophily, weighted=args.weighted, directed=args.directed, path_in=args.graph, path_out=args.output, fformat=args.format, numb_docs=args.docs, docs_path=args.docs_folder, args=args)
        topic_model(numb_topics=args.topics, numb_docs=args.docs, virality=args.virality,  path_in=args.docs_folder, path_out=args.output, fformat=args.format)
        diffusion_model(numb_steps=args.steps, actives_perc=args.actives, path_out=args.output, fformat=args.format)
    finally:
        cleaning()



if __name__ == "__main__":
    args = parse_args()
    main(args)
