# /network/network_model.py
# Abstract class defining the Network models
import pathlib
import json
import pickle
from tqdm import tqdm
from abc import ABC
import networkx as nx


class NetworkModel(ABC):
    '''
    Abstract class for network models

    Attributes
    ----------
    - graph : dict
    - info : dict

    Methods
    -------
    gformat : static
        formats the graph as dict from networkx format
    save_model_attr : concrete
        saves all the attributes
    save_model_class : concrete
        saves all the class in pickle file

    Notes
    -----
    Hidden attributes names has to start with "Hidden_"
    '''

    def __init__(self):
        self.graph = {}
        self.info = {}


    @staticmethod
    def gformat(nx_obj, directed=False):
        '''
        Formatting NetworkX graph to a Python dictionary format

        Parameters
        ----------
        - nx_obj : networkx object
          NetworkX instance of the graph

        Returns
        -------
        - [dict] Python3 dictionary of the graph where:
          key <- tuple which describes the link (node_1, node_2)
          value <- int weight of the link
        '''
        #graph_adj = nx.adjacency_matrix(nx_obj).toarray()
        G = {}
        print('Formatting graph:')
        if directed:
            for edge in tqdm(nx_obj.edges()):
                G[(edge[0],edge[1])] = 1
        else:
            #print(nx_obj.edges())
            for edge in tqdm(nx_obj.edges()):
                #print(edge)
                G[(edge[0],edge[1])] = 1
                G[(edge[1],edge[0])] = 1

        return G
