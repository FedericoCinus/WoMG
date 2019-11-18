# /network/tn.py
# Implementation of the Topic-aware Network model
import os
import womg_core
import random
import pathlib
import collections
import networkx as nx
import numpy as np
from sklearn.decomposition import NMF
from tqdm import tqdm, tqdm_notebook
from womg_core.network.tlt_network_model import TLTNetworkModel
from womg_core.utils.utility_functions import read_edgelist
from womg_core.utils.distributions import random_powerlaw_vec





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
    - _nx_obj : NetworkX object
        networkx instance of the input network
    - godNode_links : dict
        dictionary containing all the links of the god node;
        god node is out connected
        to all the nodes but does not have in connections;
        god node index (id) is -1; format will be:
        key <- (-1, node id) [all int]
        value <- link weight [int]
    - _numb_topics : int
        dimension of the interests and influence vectors
    - _fast : bool
        flag for defining the chosen method for interests generation.
        if True the fastest (random) method is chosen

    Methods
    -------
    - set_graph()
    - set_godNode_links()
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

    def __init__(self, numb_topics, homophily,
                 weighted, directed,
                 graph_path,
                 interests_path,
                 gn_strength,
                 infl_strength,
                 p, q, num_walks,
                 walk_length,
                 window_size, workers, iiter,
                 beta, norm_prior,
                 alpha_value, beta_value,
                 prop_steps=5000,
                 progress_bar=False,
                 seed=None
                 ):
        super().__init__()
        self.users_interests = {}
        self.users_influence = {}
        self._numb_topics = numb_topics
        self._homophily = homophily
        self._weighted = weighted
        self._directed = directed
        self._graph_path = graph_path
        self._interests_path = interests_path
        self.godNode_links = {}
        self._godNode_strength = gn_strength
        self._infl_strength = infl_strength
        self._rand = 16 -15.875*self._homophily
        #self.node2vec = Node2VecWrapper(p, q, num_walks, ...)
        self._p = p
        self._q = q
        self._num_walks = num_walks
        self._walk_length = walk_length
        self._window_size = window_size
        self._workers = workers
        self._iiter = iiter
        self._beta = beta
        self._norm_prior = norm_prior
        self._alpha_value = alpha_value
        self._beta_value = beta_value
        if progress_bar:
            self._progress_bar = tqdm_notebook
        else:
            self._progress_bar = tqdm
        self._prop_steps = prop_steps
        self._seed = seed



    def network_setup(self, int_mode):
        '''
        - Sets the graph atribute using set_graph() method
        - Sets the info attribute using set_info() method
        - Sets the godNode_links attribute using set_godNode_links() method
        - Sets the interests vectors using set_interests() method
        - Sets the influnce vecotrs using set_influence() mehtod
        - Sets the new graph weights using update_weights() method

        Notes
        -----
        See each method docstring for details
        '''
        if self._graph_path == None:
            print('No graph path provided \n DEMO Mode: generating cascades in les miserables network')
            self._graph_path = pathlib.Path(os.path.abspath(womg_core.__file__)[:-21]) / "womgdata" / "graph" / "lesmiserables" / "lesmiserables_edgelist.txt"
            self._nx_obj, self.mapping  = read_edgelist(self,path=self._graph_path, weighted=False, directed=False)
        else:
            self._graph_path = pathlib.Path(self._graph_path)
            self._nx_obj, self.mapping = read_edgelist(self, path=self._graph_path, weighted=self._weighted, directed=self._directed)
        self.set_graph()
        self.set_godNode_links()
        self.set_interests(int_mode)
        self.set_influence()
        #print('updating weights')
        self.graph_weights_vecs_generation()


    def set_graph(self):
        '''
        Sets the graph attribute formatting the networkx instance with gformat()
        method of the superclass
        '''
        if isinstance(self._nx_obj, nx.classes.graph.Graph):
            self.graph = self.gformat(self._nx_obj, directed=self._directed)
            self.set_info()
        else:
            print('Not a networkx readable object')


    def set_info(self):
        '''
        Sets graph info dictionary attribute using networkx graph-instance description
        '''
        infos = nx.info(self._nx_obj) + '\nDirected: '+str(nx.is_directed(self._nx_obj))
        infos = infos.split()
        self.info['type'] = infos[2]
        self.info['numb_nodes'] = infos[6]
        self._numb_nodes = int(infos[6])
        self.info['numb_edges'] = infos[10]
        if infos[2] == 'MultiDiGraph':
            self.info['aver_in_degree'] = infos[14]
            self.info['aver_out_degree'] = infos[18]
            self.info['directed'] = infos[20]
        else:
            self.info['aver_degree'] = infos[13]
            self.info['directed'] = infos[15]



    def set_godNode_links(self, weight=1, nodes=None):
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
        tn_instance.set_godNode_links() :
        all the links weights (from godNode to each node) are set to 1
        '''
        if nodes is None:
            print('Setting god node')
            for node in self._progress_bar(self._nx_obj.nodes()):
                self.godNode_links[(-1, node)] = weight
                self.graph.update(self.godNode_links)
        if isinstance(nodes, int):
            self.godNode_links[(-1, nodes)] = weight
            self.graph.update(self.godNode_links)
        if isinstance(nodes, collections.Iterable):
            for node in nodes:
                self.godNode_links[(-1, node)] = weight
                self.graph.update(self.godNode_links)

    def set_interests(self, int_mode):
        '''
        Creates interests vectors (numb_topics dimension) for each node and
        save them in the users_interests class attribute.

        Parameters
        ----------
        - method : string
          name of the method for creating interests vectors
        '''
        if int_mode == 'rand':
            print('Random generation of interests:')
            self.random_interests()
        if int_mode == 'n2i':
            print('Generating interests from graph in ')
            self._q = np.exp(4.60517*self._homophily)
            self._p = np.exp(-4.60517*self._homophily)
            prior = 'half_norm' if self._norm_prior else 'beta'
            emb = n2i_nx_graph(
                        nx_graph=self._nx_obj,
                        window_size=self._window_size,
                        walk_length=self._walk_length,
                        num_walks=self._num_walks,
                        dimensions=self._numb_topics,
                        p=self._p,
                        q=self._q,
                        beta=self._beta,
                        prior=prior,
                        alpha_value=self._alpha_value,
                        beta_value=self._beta_value,
                        seed=self._seed
                   )
            for node in self._nx_obj.nodes():
                self.users_interests[node] = emb[node]

        if int_mode == 'prop_int':
            self.propag_interests()

        if int_mode == 'nmf':
            self.nmf_interests()

        if int_mode == 'load' or self._interests_path != None:
            print('Loading interests from: ', self._interests_path)
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
            for node in self._nx_obj.nodes():
                self.users_influence[node] = [0. for _ in range(self._numb_topics)]
        else:
            fitness = 1 + np.random.pareto(a=self._infl_strength, size=self._nx_obj.number_of_nodes())
            for node in self._nx_obj.nodes():
                self.users_influence[node] = fitness[node] * self.users_interests[node]
        '''
        random_powerlaw_vec(gamma=self._rho, self._numb_topics)
        # rescaling infleunce importance
        norm_avg = 0.
        for node in self._nx_obj.nodes():
            norm_avg += np.linalg.norm(self.users_interests[node])/self._numb_nodes
        scale_fact = self._infl_strength*norm_avg
        # setting influence vec
        for node in self._nx_obj.nodes():
            influence_vec = self.node2influence(scale_fact)
            self.users_influence[node] = influence_vec
        '''

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
            for node in self._nx_obj.nodes():
                self.users_interests[node] = np.random.rand(self._numb_topics)


    def propag_interests(self):
        '''
        generates interests with the propagation method starting from "n" nodes
        '''
        # INITIALIZE
        for i in list(self._nx_obj.nodes):
            interests = np.random.dirichlet(np.ones(self._numb_topics)*1./self._numb_topics)
            self.users_interests[i] = interests

        # SELECT INFLUENCERS
        # start with a random node
        influencers = [np.random.choice(self._nx_obj.nodes)]

        for i in range(14):
            # calculate distances to current influencers
            sp = {i: nx.shortest_path(self._nx_obj, i) for i in influencers}
            distances = np.array([list(len(sp[j][i]) for j in influencers) for i in self._nx_obj.nodes()])
            # select the node fartest from all influencers
            influencers.append(distances.min(axis=1).argmax())

        # PROPAGATION STEP
        for c in range(self._prop_steps):
            i = np.random.choice(self._nx_obj.nodes())
            interests_i = self.users_interests[i]
            lr = 0.5 if i in influencers else 0.01
            #lr = 0.1
            for j in list(self._nx_obj.neighbors(i)):
                if j in influencers:
                    continue
                interests_j = self.users_interests[j]
                interests_j += interests_i * lr
                interests_j /= interests_j.sum()
                self.users_interests[j] = interests_j



    def overlap_generator(self):
        """
        Function to generate a neighbourhood overlap matrix (second-order proximity matrix).
        :param G: Graph object.
        :return laps: Overlap matrix.
        """
        G = self._nx_obj
        degrees = nx.degree(G)
        sets = {node:set(G.neighbors(node)) for node in nx.nodes(G)}
        laps = np.array([[float(len(sets[node_1].intersection(sets[node_2])))/(float(degrees[node_1]*degrees[node_2])**0.5) if node_1 != node_2 else 0.0 for node_1 in nx.nodes(G)] for node_2 in nx.nodes(G)],dtype = np.float64)
        return laps

    def nmf_interests(self, eta=64.):
        '''
        Generates interests according to non-negative matrix factorization
        method

        Parameters
        ----------

        Returns
        -------

        '''
        print('Generating interests with nmf ..')
        A = nx.to_numpy_matrix(self._nx_obj)
        S_0 = self.overlap_generator()
        R = np.random.rand(self._nx_obj.number_of_nodes(), self._nx_obj.number_of_nodes())

        S = eta*S_0 + A + self._rand*R
        model = NMF(n_components=self._numb_topics, init='random', random_state=self._seed)
        W = model.fit_transform(S)

        for node in self._nx_obj.nodes():
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
        for link in self.graph.keys():
            # god node
            if link[0] == -1:
                out_influence_vec = np.array([self._godNode_strength for i in range(self._numb_topics)])
                in_interest_vec = self.users_interests[link[1]]
                self.set_link_weight(link, out_influence_vec + in_interest_vec)
            else:
                out_influence_vec = self.users_influence[link[0]]
                in_interest_vec = self.users_interests[link[1]]
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


    def load_interests(self):
        '''
        Loads interests vector from path
        '''
        with open(self._interests_path, 'r') as f:
            for _ in f.readlines():
                node, interests = _.split(' ', 1)[0], _.split(' ', 1)[1]
                self.users_interests[int(node)] = np.array(eval(interests[:-1]))
