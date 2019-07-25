# /network/tn.py
# Implementation of the Topic-aware Network model
import random
import pathlib
import collections
import networkx as nx
import numpy as np
from tqdm import tqdm, tqdm_notebook
from sklearn.decomposition import NMF
from n2i.__main__ import n2i_main, n2i_nx_graph
from node2vec_git.src.node2vec_main import node2vec_main
from network.tlt_network_model import TLTNetworkModel
from utils.utility_functions import read_edgelist



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
                 weighted, directed, graph_path,
                 gn_strength,
                 infl_strength,
                 p, q, num_walks,
                 walk_length, dimensions,
                 window_size, workers, iiter,
                 beta, norm_prior,
                 progress_bar=False):
        super().__init__()
        self.users_interests = {}
        self.users_influence = {}
        self._numb_topics = numb_topics
        self._homophily = homophily
        self._weighted = weighted
        self._directed = directed
        self._graph_path = graph_path
        self.godNode_links = {}
        self._godNode_strength = gn_strength
        self._infl_strength = infl_strength
        #self.node2vec = Node2VecWrapper(p, q, num_walks, ...)
        self._p = p
        self._q = q
        self._num_walks = num_walks
        self._walk_length = walk_length
        self._dimensions = dimensions
        self._window_size = window_size
        self._workers = workers
        self._iiter = iiter
        self._beta = beta
        self._norm_prior = norm_prior
        if progress_bar:
            self._progress_bar = tqdm_notebook
        else:
            self._progress_bar = tqdm


    def network_setup(self, fast):
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
            self._graph_path = pathlib.Path("../") / "data" / "graph" / "lesmiserables" / "lesmiserables_edgelist.txt"
            self._nx_obj = read_edgelist(self,path=self._graph_path, weighted=False, directed=False)
        else:
            self._graph_path = pathlib.Path(self._graph_path)
            self._nx_obj = read_edgelist(self, path=self._graph_path, weighted=self._weighted, directed=self._directed)
        self.set_graph()
        self.set_godNode_links()
        self.set_interests(fast)
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

    def set_interests(self, fast):
        '''
        Creates interests vectors (numb_topics dimension) for each node and
        save them in the users_interests class attribute.

        Parameters
        ----------
        - method : string
          name of the method for creating interests vectors
        '''

        if fast:
            print('Fast generation of interests:')
            self.random_interests()
        else:
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
                        prior=prior
                   )
            for node in range(self._numb_nodes):
                self.users_interests[node] = emb[node]



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
        for node in self._nx_obj.nodes():
            norm_avg += np.linalg.norm(self.users_interests[int(node)])/self._numb_nodes
        scale_fact = self._infl_strength*norm_avg
        # setting influence vec
        for node in self._nx_obj.nodes():
            if method == 'node2influence':
                influence_vec = self.node2influence(scale_fact)
                self.users_influence[node] = influence_vec

    def node2interests(self,  transl=True, norm=False):
        '''
        Create interests vector for each node using node2vec algorithm and
        directly saves interests vectors in attribute 'users_interests'.

        3 steps:
            1. finding node2vec embeddings
            2. translation to positive axes of the embeddings
            3. reduction with NMF to _numb_topics dimensions

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
          of features (or dimensions) is equal to the number of topics (_numb_topics)
        - Method arg exists for a generalization aim: other method can be implemented
        - p is fixed to 1, while q ranges from 0.25 (max homophily) to 4 (min homophily)
        '''
        # Final matrix
        M = []
        # Node2Vec

        self._q = np.exp(4.60517*self._homophily)
        self._p = np.exp(-4.60517*self._homophily)
        interests_model = node2vec_main(weighted=self._weighted,
                                        graph=self._graph_path,
                                        directed=self._directed,
                                        p=self._p, q=self._q,
                                        num_walks=self._num_walks,
                                        walk_length=self._walk_length,
                                        dimensions=self._dimensions,
                                        window_size=self._window_size,
                                        workers=self._workers,
                                        iiter=self._iiter)
        #interests_model = node2vec.fit(window=10, min_count=1)
        #print(interests_model.wv.vocab)

        # Translation
        if transl:
            # translation constant
            minim = 0.
            for i in interests_model.wv.vocab:
                if min(interests_model.wv[str(i)]) < minim:
                    minim = min(interests_model.wv[str(i)])
            ##
            print("Computing interest vectors: ")
            for node in self._progress_bar(sorted(interests_model.wv.vocab)):
                self.users_interests[int(node)] = np.array([])
                for topic in range(self._dimensions):
                    self.users_interests[int(node)] = np.append(self.users_interests[int(node)],
                                                                      interests_model.wv[node][topic] + abs(minim))
                # Normalization
                if norm:
                    self.users_interests[int(node)] = self.users_interests[int(node)] / self.users_interests[int(node)].sum()
                M.append(self.users_interests[int(node)])
        # NO Translation
        else:
            for node in sorted(interests_model.wv.vocab):
                self.users_interests[int(node)] = np.array([])
                for topic in range(self._dimensions):
                    self.users_interests[int(node)] = np.append(self.users_interests[int(node)],
                                                                  interests_model.wv[node][topic])
                M.append(self.users_interests[int(node)])

        # NMF Reduction
        print('Reducing dimensions from ', self._dimensions,' to ', self._numb_topics)
        nmf = NMF(n_components=self._numb_topics, random_state=42, max_iter=1000)
        right = nmf.fit(M).components_
        left = nmf.transform(M)
        for node, index in zip(sorted(interests_model.wv.vocab), range(self._numb_nodes)):
            self.users_interests[int(node)] = left[int(index)]
            #print(self.users_interests[int(node)])


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
            for node in range(self._numb_nodes):
                self.users_interests[int(node)] = np.random.rand(self._numb_topics)


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
