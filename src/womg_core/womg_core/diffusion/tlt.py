 # /Diffusion/tlt.py
# Implementation of TLT model
import pathlib
import pickle
import numpy as np
from tqdm import tqdm, tqdm_notebook

from womg_core.diffusion.diffusion_model import DiffusionModel
from womg_core.utils.saver import TxtSaver


class TLT(DiffusionModel):
    '''
    Implementing Topic-aware Linear Threshold model

    Attributes
    ----------
    _numb_steps : int
    _numb_nodes : int
    _numb_docs : int
    _numb_topics : int
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
                 progress_bar=False, single_activator=False):
        super().__init__()
        self.network_model = network_model
        self.topic_model = topic_model
        self.saver = TxtSaver(path_out)
        self.new_active_nodes = {}
        self.active_nodes = {}
        self.inactive_nodes = {}
        self._propagations = []
        self._numb_nodes = 0
        self._numb_docs = 0
        self._numb_topics = 0
        self._single_activator = single_activator
        print("single_activator:", single_activator)
        if progress_bar:
            self._progress_bar = tqdm_notebook
        else:
            self._progress_bar = tqdm



    def diffusion_setup(self):
        '''
        Sets all the attributes of the current and super class
        '''
        self._numb_nodes = int(self.network_model.info['numb_nodes'])
        self._numb_docs = int(self.topic_model.numb_docs)
        self._numb_topics = int(self.topic_model.numb_topics)
        self._stall_count = {}
        for i in range(self._numb_docs):
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
        self.saver.save_propagation(self.propagations, step)

        if step == 0:
            for item in range(self._numb_docs):
                new_active_nodes = self.active_nodes[item]
                self.update_sets(item, new_active_nodes)
        for item in self._progress_bar(range(self._numb_docs)):
            new_active_nodes = set()
            if self.active_nodes[item] != set() or None:
                for node in self.inactive_nodes[item]:
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
        #print(self.topic_model.viralities[item])
        threshold = 1/self.topic_model.viralities[item]
        # calculating the sum over active nodes on item linked with user

        v_sum = np.zeros(self._numb_topics)
        node_check = False

        if self.network_model._directed:
            neighbors = [u for u, _v in list(self.network_model._nx_obj.in_edges(node))]
        else:
            neighbors = [v for _u, v in list(self.network_model._nx_obj.edges(node))]

        for v in neighbors:
            if v != node and  v in self.active_nodes[item]:
                v_sum += np.array(self.network_model.graph[(v, node)])
                node_check = True
        if node_check:
            z_sum = np.dot(self.topic_model.items_descript[item], v_sum)
            #print(1/(np.exp(- z_sum)+1), threshold)
            return (1/(np.exp(- z_sum)+1)) >= threshold
        else:
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
            #print('item ', item)
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
        for item in range(self._numb_docs):
            self.active_nodes[item] = self.godNode_influence_config(item)
            if self.active_nodes[item] == set():
                self._stall_count[item] += 1
            self.inactive_nodes[item] = set(self.network_model._nx_obj.nodes())

            self.new_active_nodes[item] = self.active_nodes[item]


    def stop_criterior(self):
        '''
        Stops the run if there are not new active nodes for given time step seq
        '''
        stall_factor = True
        for item in self._stall_count.keys():
            stall_factor *= (self._stall_count[item] >= 1)
        return stall_factor


    def godNode_influence_config(self, item):
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
        threshold = 1/self.topic_model.viralities[item]
        max_interested = -np.inf
        max_v = None
        for u, v in self.network_model.godNode_links:
            curr_weight = self.network_model.graph[(u, v)]
            z_sum = np.dot(self.topic_model.items_descript[item], curr_weight)
            #print(1/(np.exp(- z_sum)+1), threshold)
            interested = (1/(np.exp(- z_sum)+1))
            if interested >= threshold:
                if self._single_activator:
                    if max_interested < interested:
                        max_interested = interested
                        max_v = v
                else:
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


    def read_class_pickle(self, model):
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
        file = pathlib.Path.cwd() / str('__'+model+'_model')
        with open(file, 'rb') as f:
            rfile = pickle.load(f)
        return rfile
