# /network/tlt_network_model.py
# Abstract class defining the Network models
import abc
from womg_core.network.network_model import NetworkModel

class TLTNetworkModel(NetworkModel):
    '''
    Abstract class for network models involved in the tlt diffusion model class


    Methods
    -------
    graph_weights_vecs_generation : abstract
        method for generating numb_topics dim vectors for each link
    validate_network_config : concrete
        validate the configuration of the class model for the tlt class

    '''

    @abc.abstractmethod
    def graph_weights_vecs_generation(self):
        '''
        Generates numb_topics dim vectors for each link and put them as
        value of the graph dict attribute of the class
        '''
        pass

    def validate_network_config(self):
        '''
        Checks the graph structure for the tlt model

        graph attribute must be a dictionary in which:
        - key : tuple
          tuple of the link
        - value : array
          numb_topics dim array of weights
        '''
        pass
