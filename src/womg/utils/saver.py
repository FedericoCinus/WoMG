import abc
import pickle
import pathlib

class Saver():
    '''
    Abstract class for defining the structure of a saver object
    '''
    def __init__(self):
        self._path = ''

    @abc.abstractmethod
    def save_diffusion(self, diffusion_model):
        '''
        Abstract method for saving a diffusion model at each step of simulation
        '''
        return

    @abc.abstractmethod
    def save_model(self, model):
        '''
        Abstract method for saving a network model or topic model
        '''
        return

    @staticmethod
    def make_output_directory(path):
        '''
        Returns the output directory path:
        - in case path arg is None -> it create a Output folder in the parent
        folder of the current path
        - in case path is given -> it check that parent folders exist
        '''
        if path in (None, ''):
            output_dir = pathlib.Path.cwd().parent / "Output"
        else:
            output_dir = pathlib.Path(path)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir


class JsonSaver(Saver):
    '''
    Concrete class for defining methods for saving in json format
    '''
    def __init__(self, path):
        super().__init__()
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
        super().__init__()
        self._path = path

    def save_users_interests(self, network_model):
        '''
        Concrete method for saving the network model interests attribute
        in a txt format: user_id_int [K-array of interests]
        '''
        output_dir = self.make_output_directory(self._path)

        filename = output_dir / str("Users_interests_sim0.txt")
        sim_numb = 0
        while pathlib.Path(filename).exists():
            sim_numb+=1
            filename = output_dir / str("Users_interests_sim" + str(sim_numb) + ".txt")

        with open(filename, "w") as f:
            for node in network_model.users_influence.keys():
                f.write(str(node) + ' ' +str(list(network_model.users_interests[node])) + '\n')


    def save_users_influence(self, network_model):
        '''
        Concrete method for saving the network model influence attribute
        in a txt format: user_id_int [K-array of influence]
        '''
        output_dir = self.make_output_directory(self._path)

        filename = output_dir / str("Users_influence_sim0.txt")
        sim_numb = 0
        while pathlib.Path(filename).exists():
            sim_numb+=1
            filename = output_dir / str("Users_influence_sim" + str(sim_numb) + ".txt")

        with open(filename, "w") as f:
            for node in network_model.users_influence.keys():
                f.write(str(node) + ' ' +str(list(network_model.users_influence[node])) + '\n')


    def save_items_descript(self, topic_model):
        '''
        Concrete method for saving the topic model items description
        in a txt format: item_id_int [K-array of probabilities]
        '''
        output_dir = self.make_output_directory(self._path)

        filename = output_dir / str("Items_description_sim0.txt")
        sim_numb = 0
        while pathlib.Path(filename).exists():
            sim_numb+=1
            filename = output_dir / str("Items_description_sim" + str(sim_numb) + ".txt")

        with open(filename, "w") as f:
            for item in topic_model.items_descript.keys():
                f.write(str(item) + ' ' +str(list(topic_model.items_descript[item])) + '\n')

    def save_topics_descript(self, topic_model):
        '''
        Concrete method for saving the topic model items description
        in a txt format: topic_id_int [V-array of probabilities]
        '''
        output_dir = self.make_output_directory(self._path)

        filename = output_dir / str("Topics_descript_sim0.txt")
        sim_numb = 0
        while pathlib.Path(filename).exists():
            sim_numb+=1
            filename = output_dir / str("Topics_descript_sim" + str(sim_numb) + ".txt")

        with open(filename, "w") as f:
            f.write(str(topic_model.topics_descript))


    def save_items_keyw(self, topic_model):
        '''
        Concrete method for saving the network model or topic model files
        in a txt format
        '''
        output_dir = self.make_output_directory(self._path)

        filename = output_dir / str("Items_keyw_sim0.txt")
        sim_numb = 0
        while pathlib.Path(filename).exists():
            sim_numb+=1
            filename = output_dir / str("Items_keyw_sim" + str(sim_numb) + ".txt")

        with open(filename, "w") as f:
            f.write(str(topic_model.items_keyw))

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
