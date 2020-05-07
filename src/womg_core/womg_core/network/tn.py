'''/network/tn.py
Implementation of the Topic-aware Network model
'''

import os
import random
import pathlib
import collections
import networkx as nx
import numpy as np
from scipy import sparse
from scipy.spatial import distance
from tqdm import tqdm, tqdm_notebook
from sklearn.decomposition import NMF
import womg_core
from womg_core.network.tlt_network_model import TLTNetworkModel
from womg_core.utils.utility_functions import read_edgelist


DEFAULT_GRAPH = pathlib.Path(os.path.abspath(womg_core.__file__).replace('/womg/__init__.py',
                             ''))/ "womgdata" / "graph" / "lesmiserables" / "lesmiserables_edgelist.txt"

class TN(TLTNetworkModel):
    '''
    Implementation of interest-influence model

    Attributes
    ----------
    - users_interests : dict
        dictionary containing the  interests vectors
        (of _numb_topics dimension) in the format:
        key <- [node id]
        value <- _numb_topics dimension array in numpy format
    - users_influence : dict
        dictionary containing the  influence vectors
        (of _numb_topics dimension) in the format:
        key <- [node id]
        value <- _numb_topics dimension array in numpy format
    - nx_obj : NetworkX object
        networkx instance of the input network
    - _numb_topics : int
        dimension of the interests and influence vectors
    - _fast : bool
        flag for defining the chosen method for interests generation.
        if True the fastest (random) method is chosen

    Methods
    -------
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

    def __init__(self, numb_topics, #pylint: disable=too-many-arguments,
                 homophily,
                 weighted, directed,
                 graph,
                 interests,
                 gn_strength,
                 infl_strength,
                 progress_bar,
                 seed,
                 ):
        super().__init__()
        self._graph = graph
        self._weighted = weighted
        self.directed = directed
        self.users_interests = {}
        self.users_influence = {}
        self._numb_topics = numb_topics
        self._homophily = homophily
        self._interests = interests
        assert gn_strength is None or np.isfinite(gn_strength)
        self._godnode_strength = gn_strength
        self._infl_strength = infl_strength
        self._rand = 16 - 15.875 * self._homophily
        if progress_bar:
            self._progress_bar = tqdm_notebook
        else:
            self._progress_bar = tqdm
        self._verbose = False
        self._seed = seed
        self.nx_obj = None
        self.mapping = None




    def network_setup(self, int_mode):
        '''
        - Sets the interests vectors using set_interests() method
        - Sets the influence vectors using set_influence() mehtod
        - Sets the new graph weights using update_weights() method

        Notes
        -----
        See each method docstring for details
        '''
        if isinstance(self._graph, nx.classes.graph.Graph):
            self.nx_obj = self._graph.copy()
            self.directed = self.nx_obj.is_directed()
            if not self.directed:
                self.nx_obj = self.nx_obj.to_directed()
            self._weighted = nx.is_weighted(self.nx_obj)
        elif self._graph is None:
            print('No graph path provided \n',
                  'DEMO Mode: generating cascades in les miserables network')
            self._graph = DEFAULT_GRAPH
            self.nx_obj, self.mapping = read_edgelist(path=self._graph,
                                                      weighted=False,
                                                      directed=False)
        else:
            self._graph = pathlib.Path(self._graph)
            self.nx_obj, self.mapping = read_edgelist(path=self._graph,
                                                      weighted=self._weighted,
                                                      directed=self.directed)
        if self._godnode_strength is not None:
            self.set_godnode_links()
        self.set_interests(int_mode)
        ##### check inf ##########################################
        for node, interests in self.users_interests.items():
            if not all(np.isfinite(interests)):
                print('ATTENTION  node: ', node, '  interests: ', interests)
        ###########################################################
        self.set_influence()
        self.graph_weights_vecs_generation()
        if self._verbose:
            print('Macroscopic homophily level: ',
                  self.homophily(),
                  ' with H=', self._homophily)



    def set_godnode_links(self, weight=1, nodes=None):
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
        tn_instance.set_godnode_links() :
        all the links weights (from godNode to each node) are set to 1
        '''
        if nodes is None:
            #print('Setting god node')
            god_node_links = []
            for node in self._progress_bar(self.nx_obj.nodes()):
                rand_num = np.abs(np.random.randn())
                god_node_links.append((-1, node, rand_num))
            self.nx_obj.add_weighted_edges_from(god_node_links)
        if isinstance(nodes, int):
            self.nx_obj.add_weighted_edges_from([(-1, nodes, weight)])
        if isinstance(nodes, collections.Iterable):
            for node in self._progress_bar(nodes):
                self.nx_obj.add_weighted_edges_from([(-1, node, weight)])


    def set_interests(self, int_mode):
        '''
        Creates interests vectors (numb_topics dimension) for each node and
        save them in the users_interests class attribute.

        Parameters
        ----------
        - method : string
          name of the method for creating interests vectors
        '''
        print('Creating interests..')
        if int_mode == 'rand':
            #print('Random generation of interests:')
            self.random_interests()

        if int_mode == 'nmf':
            self.nmf_interests()

        if int_mode == 'load' or self._interests is not None:
            if isinstance(self._interests, dict):
                print('Loading interests')
            else:
                print('Loading interests from: ', self._interests)
            self.load_interests()


    def set_influence(self):
        '''
        Sets influence vectors for all nodes

        Parameters
        ----------
        - method : string
          name of the method for creating interests vectors
        '''

        if self._infl_strength is None:
            for node in self.nx_obj.nodes():
                self.users_influence[node] = [0. for _ in range(self._numb_topics)]
        else:
            fitness = 1 + np.random.pareto(a=self._infl_strength,
                                           size=self.nx_obj.number_of_nodes())
            if np.any(np.isinf(fitness)):
                print('Wrong parameter infl strenght')
            fitness = np.minimum(fitness, 10E20)
            for node in self.nx_obj.nodes():
                self.users_influence[node] = fitness[node] * self.users_interests[node]

        if self._godnode_strength is not None:
            self.users_influence[-1] = np.ones(self._numb_topics)

        for node, value in self.users_influence.items():
            if not all(np.isfinite(value)):
                print('ATTENTION node: ', node, ' influences: ', value)


    def random_interests(self, norm=True):
        '''
        Create interests vector for each node using a random uniform probability density
        function and directly saves interests vectors in attribute users_interests

        Parameters
        ----------
        - norm : bool
        if True interests vectors are normalized

        '''
        if norm:
            for node in self.nx_obj.nodes():
                self.users_interests[node] = np.random.rand(self._numb_topics)

    @staticmethod
    def overlap_generator(A):
        """
        Generate the second order overlap from a sparse adjacency matrix A.
        """
        aat = A.dot(A.T)
        d = aat.diagonal()
        ndiag = sparse.diags(d, 0)
        n = np.sqrt(ndiag.dot(aat > 0).dot(ndiag))
        n.data[:] = 1./n.data[:]
        return aat.multiply(n) #- sparse.identity(aat.shape[0])



    def nmf_interests(self, eta=64.):
        '''
        Generates nodes interests using NMF
        '''
        #beta = self._homophily
        A = nx.adjacency_matrix(self.nx_obj)
        S_0 = self.overlap_generator(A)
        R = np.random.rand(self.nx_obj.number_of_nodes(), self.nx_obj.number_of_nodes())
        #S = beta*(S_0 + A + sparse.identity(A.shape[0])) + (1-beta)*R
        eta = 64.
        S = eta*S_0 + A + self._rand*R
        model = NMF(n_components=self._numb_topics, init='nndsvd', random_state=self._seed)
        W = model.fit_transform(S)
        if not np.all(np.isfinite(W)):
            print('ATTENTION W contains infinites')

        for node in self.nx_obj.nodes():
            self.users_interests[node] = W[node]

    def node2influence(self, scale_fact, alpha_max=10):
        '''
        Creates influence vectors (numb_topics dimension) from dirichlet
        distribution over topics for a single node

        Parameters
        ----------
        - alpha_max : int
          highest value in the dirichlet weight vec for a random entry
        '''
        dirich_weight_vec = [1 for topic in range(self._numb_topics)]
        dirich_weight_vec[random.randint(0, self._numb_topics-1)] = alpha_max
        influence_vec = np.random.dirichlet(dirich_weight_vec)
        return scale_fact*influence_vec


    def graph_weights_vecs_generation(self):
        '''
        Creates weights vectors (numb_topcis dimension) for each direct link
        and update the graph attribute; a link weight is defined as:

        w_(i,j) = intersts_j + influence_i

        Parameters
        ----------
        - god_node_weight : int
          scalar value for each entry of the weight vector involving the godNode
        '''
        for u, v in list(self.nx_obj.edges()):
            # god node
            if u == -1:
                out_influence_vec = self.users_influence[u]
                in_interest_vec = self.users_interests[v]
                self.set_link_weight((u, v),
                                     in_interest_vec + self._godnode_strength * out_influence_vec)
            else:
                out_influence_vec = self.users_influence[u]
                in_interest_vec = self.users_interests[v]
                self.set_link_weight((u, v), out_influence_vec + in_interest_vec)


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
        u, v = link
        if not all(np.isfinite(new_weight)):
            print('ATTENTION link ', link, ' weight: ', new_weight)
        self.nx_obj[u][v]['weight'] = new_weight


    def load_interests(self, sep=','):
        '''
        Loads interests vector from path.
        Format: "node int1,int2,int3.."
        '''
        if isinstance(self._interests, dict):
            self.users_interests = self._interests
        else:
            with open(self._interests, 'r') as file:
                for line in file.readlines():
                    node, interests = line.split(' ')[0], line.split(' ')[1]
                    node_interests = [float(entry) for entry in interests[:-1].split(sep)]
                    self.users_interests[int(node)] = np.array(node_interests)


    ################# Analysis

    def sim_in(self):
        '''Returns average interests similarity between connected nodes
        '''
        sims = []
        for i in self.nx_obj.nodes:
            for j in list(self.nx_obj.neighbors(i)):
                sims.append(1 - distance.cosine(self.users_interests[i], self.users_interests[j]))
        return np.mean(sims)

    def select_notedge(self):
        '''Returns tuple of not connected nodes
        '''
        node1 = np.random.choice(self.nx_obj.nodes())
        node2 = np.random.choice(self.nx_obj.nodes())

        while (node1, node2) in self.nx_obj.edges or node1 == node2:
            node1 = np.random.choice(self.nx_obj.nodes())
            node2 = np.random.choice(self.nx_obj.nodes())
        return node1, node2

    def sim_out(self, samples):
        '''Returns average interests similarity between of not connected nodes
        samples times
        '''
        sims_out = []
        for _ in range(samples):
            i, j = self.select_notedge()
            sims_out.append(1 - distance.cosine(self.users_interests[i], self.users_interests[j]))
        return np.mean(sims_out)

    def homophily(self, numb_not_edges_tested=10000):
        '''Returns the cosine similarity homophily ratio measure
        '''
        return self.sim_in() / self.sim_out(numb_not_edges_tested)
