from abc import ABC

class Saver(ABC):
    '''
    Abstract class for defining the structure of a saver object
    '''
    def __init__(self):
        self._path = ''

    @abstractmethod
    def save_diffusion(self, diffusion_model):
        '''
        Abstract method for saving a diffusion model at each step of simulation
        '''
        return

    @abstractmethod
    def save_model(self, model):
        '''
        Abstract method for saving a network model or topic model
        '''
        return



class JsonSaver(Saver):
    '''
    Concrete class for defining methods for saving in json format
    '''
    def __init__(self, path):
        super().__init()
        self._path = path

    def save_diffusion(self, diffusion_model):
        '''
        Concrete method for saving the diffusion model or files
        in a Json format
        '''
        return

    def save_model(self, model):
        '''
        Concrete method for saving the network model or topic model files
        in a Json format
        '''
        return


class TxtSaver(Saver):
    '''
    Concrete class for defining methods for saving in json format
    '''
    def __init__(self, path):
        super().__init()
        self._path = path

    def save_users_interests(self, netwrok_model):
        '''
        Concrete method for saving the network model interests attribute
        in a txt format: user_id_int [k-array of interests]
        '''
        return

    def save_users_influence(self, network_model):
        '''
        Concrete method for saving the network model influence attribute
        in a txt format: user_id_int [k-array of influence]
        '''
        return

    def save_items_descript(self, topic_model):
        '''
        Concrete method for saving the diffusion model or files
        in a txt format
        '''
        return

    def save_items_keyw(self, topic_model):
        '''
        Concrete method for saving the network model or topic model files
        in a txt format
        '''
        return

    def save_propagations(self, diffusion_model):
        '''
        Concrete method for saving the network model or topic model files
        in a txt format
        '''
        return



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
'''
