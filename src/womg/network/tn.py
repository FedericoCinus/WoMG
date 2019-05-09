# /network/tn.py
# Implementation of the Topic-aware Network model
import random
import pathlib
import collections
import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import NMF
from womg.node2vec_git.src.node2vec_main import node2vec_main
from womg.network.tlt_network_model import TLTNetworkModel
from womg.utils.utility_functions import read_edgelist



class TN(TLTNetworkModel):
    '''
    Implementation of interest-influence model

    Attributes
    ----------
    - interests_influence : dict
      dictionary containing the influence and interests vectors
      (of Hidden_numb_topics dimension) in the format:
      key <- [node id, 'int'] (for interest) or [node id, 'infl'] (for influence)
      value <- Hidden_numb_topics dimension array in numpy format
    - Hidden_nx_obj : NetworkX object
      networkx instance of the input network
    - Hidden_godNode : dict
      dictionary containing all the links of the god node that is out connected
      to all the nodes but does not have in connections;
      god node index (id) is -1; format will be:
      key <- (-1, node id) [all int]
      value <- link weight [int]
    - Hidden_numb_topics : int
      dimension of the interests and influence vectors

    Methods
    -------
    - set_graph()
    - set_godNode()
    - set_interests()
    - set_influence()
    - node2interests(): generates realistic latent interests of the nodes starting
      from the fixed given network
    - graph_weights_vecs_generation(): updates network weights using interests
      and influence vecs

    Notes
    -----
    This class implementation inherits from the abstract class networkModel()

    References
    ----------

    '''

    def __init__(self, numb_topics, homophily, weighted, directed, path_in=None, god_node=True, numb_docs=None, docs_path=None, args=None):
        super().__init__()
        self.interests_influence = {}
        self.Hidden_numb_topics = numb_topics
        self.Hidden_homophily = homophily
        self.Hidden_weighted = weighted
        self.Hidden_directed = directed
        self.Hidden_path_in = path_in
        self.Hidden_godNode = {}
        self.Hidden_god_node = god_node
        self.Hidden_args = args
        self.network_setup()
        self.save_model_class()


    def network_setup(self):
        '''
        - Sets the graph atribute using set_graph() method
        - Sets the info attribute using set_info() method
        - Sets the _godNode attribute using set_godNode() method
        - Sets the interests vecotrs using set_interests() method
        - Sets the influnce vecotrs using set_influence() mehtod
        - Sets the new graph weights using update_weights() method

        Notes
        -----
        See each method docstring for details
        '''
        if self.Hidden_path_in == None:
            print('No graph path provided \n DEMO Mode: generating cascades in les miserables network')
            self.Hidden_path_in = pathlib.Path.cwd() / "Input" / "Graph" / "lesmiserables_edgelist.txt"
            self.Hidden_nx_obj = read_edgelist(self,path=self.Hidden_path_in, weighted=False, directed=False)
        else:
            self.Hidden_path_in = pathlib.Path(self.Hidden_path_in)
            self.Hidden_nx_obj = read_edgelist(self, path=self.Hidden_path_in, weighted=self.Hidden_weighted, directed=self.Hidden_directed)
        self.set_graph()

        if self.Hidden_god_node:
            self.set_godNode()
        self.set_interests()
        self.set_influence()
        #print('updating weights')
        self.graph_weights_vecs_generation()


    def set_graph(self):
        '''
        Sets the graph attribute formatting the networkx instance with gformat()
        method of the superclass
        '''
        if isinstance(self.Hidden_nx_obj, nx.classes.graph.Graph):
            self.graph = self.gformat(self.Hidden_nx_obj, directed=self.Hidden_directed)
            self.set_info()
        else:
            print('Not a networkx readable object')


    def set_info(self):
        '''
        Sets graph info dictionary attribute using networkx graph-instance description
        '''
        infos = nx.info(self.Hidden_nx_obj) + '\nDirected: '+str(nx.is_directed(self.Hidden_nx_obj))
        infos = infos.split()
        self.info['type'] = infos[2]
        self.info['numb_nodes'] = infos[6]
        self.Hidden_numb_nodes = int(infos[6])
        self.info['numb_edges'] = infos[10]
        if infos[2] == 'MultiDiGraph':
            self.info['aver_in_degree'] = infos[14]
            self.info['aver_out_degree'] = infos[18]
            self.info['directed'] = infos[20]
        else:
            self.info['aver_degree'] = infos[13]
            self.info['directed'] = infos[15]



    def set_godNode(self, weight=1, nodes=None):
        '''
        Sets the godNode's links weights in the graph

        Parameters
        ----------
        weight : int
            scalar weight for the link/links
            (Default 1)

        nodes : iterable/int
            nodes indexes or node index on which one wants to change link weights
            if None is given all link weights (from godNode to each node) is changed
            (Defalut None)

        Example:
        tn_instance.set_godNode() :
        all the links weights (from godNode to each node) are set to 1
        '''
        if nodes is None:
            print('Setting god node')
            for node in tqdm(self.Hidden_nx_obj.nodes()):
                self.Hidden_godNode[(-1, node)] = weight
                self.graph.update(self.Hidden_godNode)
        if isinstance(nodes, int):
            self.Hidden_godNode[(-1, nodes)] = weight
            self.graph.update(self.Hidden_godNode)
        if isinstance(nodes, collections.Iterable):
            for node in nodes:
                self.Hidden_godNode[(-1, node)] = weight
                self.graph.update(self.Hidden_godNode)

    def set_interests(self, method='node2interests'):
        '''
        Creates interests vectors (numb_topics dimension) for each node and
        save them in the interests_influence class attribute.

        Parameters
        ----------
        - method : string
          name of the method for creating interests vectors
        '''
        print('Generating interests')
        if method == 'node2interests':
            self.node2interests(norm=False)


    def set_influence(self, method='node2influence'):
        '''
        Sets influence vectors for all nodes

        Parameters
        ----------
        - method : string
          name of the method for creating interests vectors
        '''
        # rescaling infleunce importance
        norm_avg = 0.
        for node in self.Hidden_nx_obj.nodes():
            norm_avg += np.linalg.norm(self.interests_influence[int(node), 'int'])/self.Hidden_numb_nodes
        scale_fact = (1-self.Hidden_homophily)*norm_avg
        # setting influence vec
        for node in self.Hidden_nx_obj.nodes():
            if method == 'node2influence':
                influence_vec = self.node2influence(scale_fact)
                self.interests_influence[node, 'infl'] = influence_vec

    def node2interests(self, transl=True, norm=False):
        '''
        Create interests vector for each node using node2vec algorithm and
        directly saves interests vectors in attribute 'interests_influence'.

        3 steps:
            1. finding node2vec embeddings
            2. translation to positive axes of the embeddings
            3. reduction with NMF to Hidden_numb_topics dimensions

        Parameters
        ----------
        - method : string
          'node2interests' uses the algorithm based on node2vec
          (see node2interests() method)
        - transl : bool
          choose if interests have to be translated to positive axes
        - norm : bool
          choose if interests have to be normalize (True) or not (False)


        Notes
        -----
        - Arrays are translated in order to have non-negative entries
        - Dimension is reduced to the number of topics using NMF, which mantains
          positivity
        - Interests are the embeddings of the node2vec algorithm whose number
          of features (or dimensions) is equal to the number of topics (Hidden_numb_topics)
        - Method arg exists for a generalization aim: other method can be implemented
        - p is fixed to 1, while q ranges from 0.25 (max homophily) to 4 (min homophily)
        '''
        # Final matrix
        M = []
        # Node2Vec

        if self.Hidden_homophily <= 0.5:
            self.Hidden_args.q = -6*self.Hidden_homophily + 4
        else:
            self.Hidden_args.q = -3./2*self.Hidden_homophily + 7./4
        self.Hidden_args.p = 1
        interests_model = node2vec_main(self.Hidden_args)
        #interests_model = node2vec.fit(window=10, min_count=1)

        # Translation
        if transl:
            # translation constant
            minim = 0.
            for i in interests_model.wv.vocab:
                if min(interests_model.wv[str(i)]) < minim:
                    minim = min(interests_model.wv[str(i)])
            ##
            print("Computing interest vectors: ")
            for node in tqdm(sorted(interests_model.wv.vocab)):
                self.interests_influence[int(node), 'int'] = np.array([])
                for topic in range(self.Hidden_numb_topics):
                    self.interests_influence[int(node), 'int'] = np.append(self.interests_influence[int(node), 'int'],
                                                                      interests_model.wv[node][topic] + abs(minim))
                # Normalization
                if norm:
                    self.interests_influence[int(node), 'int'] = self.interests_influence[int(node), 'int'] / self.interests_influence[int(node), 'int'].sum()
                M.append(self.interests_influence[int(node), 'int'])
        # NO Translation
        else:
            for node in sorted(interests_model.wv.vocab):
                self.interests_influence[int(node), 'int'] = np.array([])
                for topic in range(self.Hidden_numb_topics):
                    self.interests_influence[int(node), 'int'] = np.append(self.interests_influence[int(node), 'int'],
                                                                  interests_model.wv[node][topic])
                M.append(self.interests_influence[int(node), 'int'])

        # NMF Reduction
        print('Reducing dimensions from ', self.Hidden_args.dimensions,' to ', self.Hidden_numb_topics)
        nmf = NMF(n_components=self.Hidden_numb_topics, random_state=42, max_iter=1000)
        right = nmf.fit(M).components_
        left = nmf.transform(M)
        for node, index in zip(sorted(interests_model.wv.vocab), range(self.Hidden_numb_nodes)):
            self.interests_influence[int(node), 'int'] = left[int(index)]
            #print(self.interests_influence[int(node), 'int'])



    def node2influence(self, scale_fact, alpha_max=10):
        '''
        Creates influence vectors (numb_topics dimension) from dirichlet
        distribution over topics for a single node

        Parameters
        ----------
        - alpha_max : int
          highest value in the dirichlet weight vec for a random entry
        '''
        dirich_weight_vec = [1 for topic in range(self.Hidden_numb_topics)]
        dirich_weight_vec[random.randint(0, self.Hidden_numb_topics-1)] = alpha_max
        influence_vec = np.random.dirichlet(dirich_weight_vec)
        return scale_fact*influence_vec


    def graph_weights_vecs_generation(self, god_node_weight=7):
        '''
        Creates weights vectors (numb_topcis dimension) for each direct link
        and update the graph attribute; a link weight is defined as:

        w_(i,j) = intersts_j + influence_i

        Parameters
        ----------
        - god_node_weight : int
          scalar value for each entry of the weight vector involving the godNode
        '''
        for link in self.graph.keys():
            # god node
            if link[0] == -1:
                out_influence_vec = np.array([god_node_weight for i in range(self.Hidden_numb_topics)])
                self.set_link_weight(link, out_influence_vec)
            else:
                out_influence_vec = self.interests_influence[(link[0], 'infl')]
                in_interest_vec = self.interests_influence[(link[1], 'int')]
                self.set_link_weight(link, out_influence_vec + in_interest_vec)


    def set_link_weight(self, link, new_weight):
        '''
        Sets the link attribute to the given weights vector arg

        Parameters
        ----------
        - link : tuple
          link in the graph in which you want to change the attribute
        - new_weight : array
          numb_topic dimension array that is going to be the new attribute of the
          link
        '''
        self.graph[link] = new_weight