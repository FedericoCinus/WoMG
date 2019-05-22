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

    def save_model_attr(self, path=None, fformat='txt', way='w'):
        '''
        Saves all network model attributes

        Parameters
        ----------
        path : string
            path in which the method will save the data,
            if None is given it will create an "Output" directory in the
            current path
        fformat : string
            defines the file format of each attribute data,
            one can choose between 'pickle' and 'json' in this string notation

         Notes
         -----
         All the attributes which start with "Hidden_" are NOT saved
        '''
        if path == None or path == '':
            output_dir = pathlib.Path.cwd().parent / "Output"
        else:
            output_dir = pathlib.Path(path)

        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        for attribute in self.__dict__.keys():
            if not str(attribute).startswith('Hidden_'):

                filename = output_dir / str("Network_" + str(attribute) + "_sim0."  + str(fformat))
                sim_numb = 0
                while pathlib.Path(filename).exists():
                    sim_numb+=1
                    filename = output_dir / str("Network_" + str(attribute) + "_sim" + str(sim_numb) + "." + str(fformat))


                if fformat == 'json':
                    with open(filename, way) as f:
                        json.dump(self.__getattribute__(str(attribute)), f)
                if fformat == 'txt':
                    with open(filename, way) as f:
                        f.write(str(self.__getattribute__(str(attribute))))
                if fformat == 'pickle':
                    with open(filename, way+'b') as f:
                        pickle.dump(self.__getattribute__(str(attribute)), f)

    def save_model_class(self):
        '''
        Saves all class in pickle format in the current directory

        Notes
        -----
        Class model file will be saved with a name that starts with "Hidden_"
        '''
        file = pathlib.Path.cwd() /  "__network_model"
        with open(file, 'wb') as f:
            pickle.dump(self, f)
