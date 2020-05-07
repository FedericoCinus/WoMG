'''
 # /Diffusion/tlt.py
# Implementation of TLT model
'''
import pathlib
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm, tqdm_notebook

from womg_core.diffusion.diffusion_model import DiffusionModel
from womg_core.utils.saver import TxtSaver


class TLT(DiffusionModel):
    '''
    Implementing Topic-aware Linear Threshold model

    Attributes
    ----------
    numb_docs : int
    numb_topics : int
    new_active_nodes : dict
    active_nodes : dict
    inactive_nodes : dict

    Methods
    -------
    diffusion_setup
        sets all the attributes of the superclass and current class
    iteration
        iterates diffusion for a single step and save results
    parameter
        valuates the tlt parameter of the diffusion process
    update_sets
        updates the active and inactive sets of nodes and current activating nodes
    setnetwork_model
        set network_model attribute to the loaded pickle object of
    tltnetwork_model class
        settopic_model
    set topic_model attribute to the loaded pickle object of
        tlttopic_model class
    set_sets
        sets the initial configuration of the active and inactive sets of nodes
    stop_criterior
        defines a way to stop the run method of the superclass
    read_class_pickle
        reads a pickle file containing a class

    Notes
    -----
    -
    '''

    def __init__(self, network_model, topic_model,
                 path_out,
                 single_activator,
                 virality_resistance,
                 progress_bar):
        super().__init__()
        self.network_model = network_model
        self.topic_model = topic_model
        self.saver = TxtSaver(path_out)
        self.new_active_nodes = {}
        self.active_nodes = {}
        self.inactive_nodes = {}
        self._propagations = []
        self.numb_nodes = None
        self.numb_docs = None
        self.numb_topics = None
        self._single_activator = single_activator
        if progress_bar:
            self._progress_bar = tqdm_notebook
        else:
            self._progress_bar = tqdm
        self._thresholds_values = []
        self._virality_resistance = virality_resistance
        self.all_propagations = None
        self._stall_count = {}


    def save_threshold_values(self, path_out):
        '''Saves all items' threshold
        '''
        with open(path_out+'/threshold_values.pickle', 'wb') as file:
            pickle.dump(self._thresholds_values, file)

    def diffusion_setup(self):
        '''
        Sets all the attributes of the current and super class
        '''
        self.numb_nodes = int(self.network_model.nx_obj.number_of_nodes())
        self.numb_docs = int(self.topic_model.numb_docs)
        self.numb_topics = int(self.topic_model.numb_topics)
        self.all_propagations = [[] for _ in range(self.numb_docs)]
        for i in range(self.numb_docs):
            self._stall_count[i] = 0
        self.set_sets()


    def iteration(self, step):
        '''
        Iterates diffusion for a single step:
        1. For each item it looks for inactive nodes and calculates the tlt
        parameter for activation.
        2. Then it updates the sets of nodes
        3. Save results contained in public attributes: new activated nodes
        '''
        #self.saver.save_propagation(self.propagations, step)
        for prop in self.propagations:
            item, node = (prop.replace('\n', '').split(' '))
            self.all_propagations[int(item)].append((step, int(node)))

        if step == 0:
            for item in range(self.numb_docs):
                new_active_nodes = self.active_nodes[item]
                self.update_sets(item, new_active_nodes)
        for item in range(self.numb_docs):
            new_active_nodes = set()
            if self.active_nodes[item] != set() or None:
                for node in self.inactive_nodes[item]:
                    if node != -1:
                        if self.parameter(item=item, node=node):
                            new_active_nodes.add(node)
                self.update_sets(item, new_active_nodes)



    def parameter(self, item, node):
        '''
        Calculate the tlt diffusion model parameter for a node on a item.
        [ Topic-aware Linear threshold model ]
             W(item, node) = topic_distr * ∑_v weight_v,node
        parameter is equal to the scalar product of topic distribution of the
        item with the sum over active nodes with out link directed to the
        given node.
        If this parameter is greater or equal to the threshold then parameter()
        returns True which means the given node will activate.
            threshold = 1/item_virality
        Parameters
        ----------
        item : int
            index of the item
        node : int
            index of the node
        Returns
        -------
        bool
            True if parameter is greater or equal to threshold, which means node
            is activating on that given item. Else False.
        '''
        # calculating the sum over active nodes on item linked with user
        v_sum = np.zeros(self.numb_topics)
        node_check = False

        if nx.is_directed(self.network_model.nx_obj):
            neighbors = [u for u, _v in list(self.network_model.nx_obj.in_edges(node)) if u != -1]
        else:
            neighbors = [v for _u, v in list(self.network_model.nx_obj.edges(node)) if v != -1]

        for v in neighbors:
            if v != node and v in self.active_nodes[item]:
                v_sum += self.network_model.nx_obj.get_edge_data(v, node)['weight']
                if not all(np.isfinite(v_sum)):
                    print('ATTENTION v_sum: ', v_sum, ' weight: ',
                          self.network_model.nx_obj.get_edge_data(v, node)['weight'],
                          ' v: ', v, ' node: ', node)
                node_check = True
        if node_check:
            z_sum = np.dot(self.topic_model.items_descript[item], v_sum)
            self._thresholds_values.append((z_sum, self.topic_model.viralities[item]))
            if not np.isfinite(z_sum):
                print('ATTENTION node: ', node, 'z_sum: ', z_sum)
            return z_sum > self._virality_resistance * self.topic_model.viralities[item]
        return False



    def update_sets(self, item, new_active_nodes):
        '''
        Updating the active, inactive and current new active sets attributes.

            The value correspondent to 'item' key of the new_active_nodes
            dictionary is set equal to the given input set (new_active_nodes).
            This set is given by the iteration method where update_sets is called.

            The inactive_nodes attribute is updating discarding the activated
            nodes calculated inside iteration()

            The active_nodes attribute is updating adding the activated
            nodes calculated inside iteration()

        Parameters
        ----------
        item : int
            index of the item
        new_active_nodes : list of int
            list of the nodes that has just activated by tlt parameter evaluation
            in parameter() method called by iteration()

        Notes
        -----
        this method is called inside the items for-loop inside iteration() method

        '''
        self.new_active_nodes[item] = new_active_nodes

        if self.new_active_nodes[item] == set():
            self._stall_count[item] += 1

        removing_list = new_active_nodes.union(self.active_nodes[item])  ### needs improvement
        if removing_list != set():
            for node in removing_list:
                self.inactive_nodes[item].discard(node)
                self.active_nodes[item].add(node)


    def set_sets(self):
        '''
        Defines the initial configuration of the sets.
            active nodes are defined with a random extraction defined in
            distributions module in utilities package

            inactive_nodes are all the nodes set

            current new activated nodes set is equal to the active one
        '''
        for item in range(self.numb_docs):
            self.active_nodes[item] = self.godnode_influence_config(item)
            if self.active_nodes[item] == set():
                self._stall_count[item] += 1
            self.inactive_nodes[item] = set(self.network_model.nx_obj.nodes())

            self.new_active_nodes[item] = self.active_nodes[item]


    def stop_criterior(self):
        '''
        Stops the run if there are not new active nodes for given time step seq
        '''
        stall_factor = True
        for item, _value in self._stall_count.items():
            stall_factor *= (self._stall_count[item] >= 1)
        return stall_factor


    def godnode_influence_config(self, item):
        '''
        Returns the activated nodes for the initial configuration for a given item;
        the god node (connected with all nodes) influences all the others for the
        given item evaluating the tlt parameter
        Parameters
        ----------
        item : int
            item index
        Returns
        -------
        list of active nodes for the given item
        '''
        actives_config = []
        max_interested = -np.inf
        max_v = None
        if self.network_model.nx_obj.is_directed():
            god_node_edges = list(self.network_model.nx_obj.out_edges(-1))
        else:
            god_node_edges = list(self.network_model.nx_obj.edges(-1))
        for u, v in god_node_edges:
            curr_weight = self.network_model.nx_obj.get_edge_data(u, v)['weight']
            z_sum = np.dot(self.topic_model.items_descript[item], curr_weight)
            self._thresholds_values.append((z_sum, self.topic_model.viralities[item]))
            if z_sum > self._virality_resistance * self.topic_model.viralities[item]:
                if self._single_activator:
                    if max_interested < z_sum:
                        max_interested = z_sum
                        max_v = v
                else:
                    if all(v == 0 for v in curr_weight):
                        print('ATTENTION nodes: ', u, v, '  curr_weight:',
                              curr_weight, ' z_sum:', z_sum)
                    actives_config.append(v)

        if self._single_activator:
            if max_v is not None:
                actives_config.append(max_v)

        return set(actives_config)


    @staticmethod
    def list_format(dictio):
        '''
        Static method for converting the dict format of the propagations
        into a list format:
            'doc; activating_node \n'
        '''
        lista = []
        for key in dictio.keys():
            for node in dictio[key]:
                lista.append(str(key) + ' ' + str(node) + '\n')
        return lista

    @property
    def propagations(self):
        '''
        When propagations is called for save it is set to the current config
        '''
        self.propagations = self.list_format(self.new_active_nodes)

        return self._propagations

    @propagations.setter
    def propagations(self, value):
        self._propagations = value

    @staticmethod
    def read_class_pickle(model):
        '''
        Read the pickle file containing a class model instance
        Parameters
        ----------
        model : string
            'topic' or 'network'
        Returns
        -------
        Loaded object of the correspondent class: TLTNetworkModel or TLTTopicModel
        '''
        filename = pathlib.Path.cwd() / str('__'+model+'_model')
        with open(filename, 'rb') as file:
            rfile = pickle.load(file)
        return rfile
