# /Diffusion/tlt.py
# Implementation of TLT model
import pathlib
import pickle
import numpy as np
from tqdm import tqdm

from womg.utils.distributions import random_initial_active_set
from womg.diffusion.diffusion_model import DiffusionModel


class TLT(DiffusionModel):
    '''
    Implementing Topic-aware Linear Threshold model

    Attributes
    ----------
    Hidden_numb_steps : int
    Hidden_numb_nodes : int
    Hidden_numb_docs : int
    Hidden_numb_topics : int
    Hidden_new_active_nodes : dict
    Hidden_active_nodes : dict
    Hidden_inactive_nodes : dict

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
    set_network_model
        set network_model attribute to the loaded pickle object of
    tlt_network_model class
        set_topic_model
    set topic_model attribute to the loaded pickle object of
        tlt_topic_model class
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

    def __init__(self, numb_steps, actives_perc, path_out, out_format='list', fformat='txt'):
        super().__init__()
        self.Hidden_numb_steps = numb_steps
        self.Hidden_numb_nodes = 0
        self.Hidden_numb_docs = 0
        self.Hidden_numb_topics = 0
        self.Hidden_actives = actives_perc
        self.Hidden_new_active_nodes = {}
        self.Hidden_active_nodes = {}
        self.Hidden_inactive_nodes = {}
        self.Hidden_path_out = path_out
        self.Hidden_out_format = out_format
        self.Hidden_fformat = fformat
        self._formatted_output = []
        self.diffusion_setup()
        self.run(self.Hidden_numb_steps)

    def diffusion_setup(self):
        '''
        Sets all the attributes of the current and super class
        '''
        self.set_network_model()
        self.set_topic_model()
        self.Hidden_numb_nodes = int(self.Hidden_network_model.info['numb_nodes'])
        self.Hidden_numb_docs = int(self.Hidden_topic_model.Hidden_numb_docs)
        self.Hidden_numb_topics = int(self.Hidden_topic_model.Hidden_numb_topics)
        self.Hidden_stall_count = [0 for i in range(self.Hidden_numb_docs)]
        self.set_sets()



    def iteration(self, step):
        '''
        Iterates diffusion for a single step:
        1. For each item it looks for inactive nodes and calculates the tlt
        parameter for activation.
        2. Then it updates the sets of nodes
        3. Save results contained in public attributes: new activated nodes
        '''
        #print(self.Hidden_stall_count)
        self.save_model_attr(step=step, fformat=self.Hidden_fformat)
        #print(self.Hidden_active_nodes)
        for item in tqdm(range(self.Hidden_numb_docs)):
            new_active_nodes = set()
            #print('ITEM: '+str(item))
            if self.Hidden_active_nodes[item] != set() or None:
                for node in self.Hidden_inactive_nodes[item]:
                    if self.parameter(item=item, node=node):
                        new_active_nodes.add(node)
                #print('item: '+str(item)+' curr new set act: '+ str(new_active_nodes))
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
        #print(self.Hidden_topic_model.viralities[item])
        threshold = 1/self.Hidden_topic_model.viralities[item]
        # calculating the sum over active nodes on item linked with user

        v_sum = np.zeros(self.Hidden_numb_topics)
        node_check = False

        if self.Hidden_network_model.Hidden_directed:
            #print("directed")
            neighbors = [i[0] for i in list(self.Hidden_network_model.Hidden_nx_obj.in_edges(node))]
        else: 
            #print("undirected")
            neighbors = [i[1] for i in list(self.Hidden_network_model.Hidden_nx_obj.edges(node))]
            #print(neighbors)


        #print(node)
        for v in neighbors:
            #print(v, node)
            #print(v in self.Hidden_active_nodes[item])
            if v != node and  v in self.Hidden_active_nodes[item]:
                
                v_sum += np.array(self.Hidden_network_model.graph[(v, node)])
                node_check = True
        if node_check:
            z_sum = np.dot(self.Hidden_topic_model.topic_distrib[item], v_sum)
            #print('value: ',(1/(np.exp(- z_sum)+1)))
            #print('threshold: ',threshold)
            return (1/(np.exp(- z_sum)+1)) >= threshold
        else:
            #print("always false")
            return False


    def update_sets(self, item, new_active_nodes):
        '''
        Updating the active, inactive and current new active sets attributes.

            The value correspondent to 'item' key of the Hidden_new_active_nodes
            dictionary is set equal to the given input set (Hidden_new_active_nodes).
            This set is given by the iteration method where update_sets is called.

            The Hidden_inactive_nodes attribute is updating discarding the activated
            nodes calculated inside iteration()

            The Hidden_active_nodes attribute is updating adding the activated
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
        self.Hidden_new_active_nodes[item] = new_active_nodes
        #print(self.Hidden_stall_count[item])
        if self.Hidden_new_active_nodes[item] == set():
            self.Hidden_stall_count[item] = self.Hidden_stall_count[item] + 1
        removing_list = new_active_nodes.union(self.Hidden_active_nodes[item])
        for node in removing_list:
            self.Hidden_inactive_nodes[item].discard(node)
            self.Hidden_active_nodes[item].add(node)



    def set_network_model(self):
        '''
        Sets the Hidden_network_model attribute to the loaded TLTNetworkModel obj
        from pickle file
        '''
        self.Hidden_network_model = self.read_class_pickle('network')

    def set_topic_model(self):
        '''
        Sets the Hidden_topic_model attribute to the loaded TLTTopicModel obj
        from pickle file
        '''
        self.Hidden_topic_model = self.read_class_pickle('topic')


    def set_sets(self):
        '''
        Defines the initial configuration of the sets.
            active nodes are defined with a random extraction defined in
            distributions module in utilities package

            Hidden_inactive_nodes are all the nodes set

            current new activated nodes set is equal to the active one
        '''
        for item in range(self.Hidden_numb_docs):
            self.Hidden_active_nodes[item] = random_initial_active_set(self, max_active_perc=self.Hidden_actives)
            self.Hidden_inactive_nodes[item] = set(self.Hidden_network_model.Hidden_nx_obj.nodes())
        self.Hidden_new_active_nodes = self.Hidden_active_nodes


    def stop_criterior(self):
        '''
        Stops the run method if inactive set is empty
        '''
        return self.Hidden_inactive_nodes == set(), ' because all nodes are active'

    def stop_criterior_2(self, stall_steps=3):
        '''
        BUG
        Stops the run if there are not new active nodes for given time step seq
        '''
        check = 0
        for item in range(len(self.Hidden_stall_count)):
            check *= (self.Hidden_stall_count[item] <= stall_steps)
        return check


    @staticmethod
    def list_format(dictio):
        '''
        Static method for converting the dict format of the propagations
        into a list format:

            'doc; activating_node \n'
        
        '''
        lista = ''
        for key in dictio.keys():
            for node in dictio[key]:
                lista += str(key) + '; ' + str(node)+'\n'
        return lista

    @property
    def formatted_output(self):
        '''
        When formatted_output is called for save it is set to the current config
        '''
        if self.Hidden_out_format == 'list':
            self.formatted_output = self.list_format(self.Hidden_new_active_nodes)
        if self.Hidden_out_format == 'dict':
            self.formatted_output = self.Hidden_new_active_nodes
        return self._formatted_output

    @formatted_output.setter
    def formatted_output(self, value):
        self._formatted_output = value


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
