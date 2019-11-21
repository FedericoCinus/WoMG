import abc
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

    @staticmethod
    def make_filename(name, output_dir, new_file=True):
        '''
        Returns the correct filename checking the file "name" is in "output_dir"
        If the file already exists it returns a filename with an incrementing
        index.

        Parameters
        ----------
        - name : str
            name of the file
        - output_dir : str
            path of the "name" file

        Returns
        -------
        Filename (str)
        '''
        filename = output_dir / str(name + "0.txt")
        sim_numb = 0

        while pathlib.Path(filename).exists():
            sim_numb+=1
            filename = output_dir / str(name + str(sim_numb) + ".txt")
        if new_file:
            return filename
        else:
            sim_numb-=1
            filename = output_dir / str(name + str(sim_numb) + ".txt")
            return filename


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
        filename = self.make_filename("Users_interests", output_dir)

        with open(filename, "w") as f:
            for node in network_model.users_influence.keys():
                f.write(str(node) + ' ' +str(list(network_model.users_interests[node])) + '\n')


    def save_users_influence(self, network_model):
        '''
        Concrete method for saving the network model influence attribute
        in a txt format: user_id_int [K-array of influence]
        '''
        output_dir = self.make_output_directory(self._path)
        filename = self.make_filename("Users_influence", output_dir)

        with open(filename, "w") as f:
            for node in network_model.users_influence.keys():
                f.write(str(node) + ' ' +str(list(network_model.users_influence[node])) + '\n')


    def save_mapping(self, network_model):
        '''
        Concrete method for saving the graph mapping
        in a txt format
        '''
        output_dir = self.make_output_directory(self._path)
        filename = self.make_filename("nodes_mapping", output_dir)

        if network_model.mapping != None:
            with open(filename, "w") as f:
                f.write(str(network_model.mapping))
        else:
            pass


    def save_items_descript(self, topic_model):
        '''
        Concrete method for saving the topic model items description
        in a txt format: item_id_int [K-array of probabilities]
        '''
        output_dir = self.make_output_directory(self._path)
        filename = self.make_filename("Items_descript", output_dir)

        with open(filename, "w") as f:
            for item in topic_model.items_descript.keys():
                f.write(str(item) + ' ' +str(list(topic_model.items_descript[item])) + '\n')

    def save_topics_descript(self, topic_model):
        '''
        Concrete method for saving the topic model items description
        in a txt format: topic_id_int [V-array of probabilities]
        '''
        output_dir = self.make_output_directory(self._path)
        filename = self.make_filename("Topics_descript", output_dir)

        if topic_model.topics_descript != {}:
            with open(filename, "w") as f:
                f.write(str(topic_model.topics_descript))
        else:
            pass


    def save_items_keyw(self, topic_model):
        '''
        Concrete method for saving the network model or topic model files
        in a txt format
        '''
        output_dir = self.make_output_directory(self._path)
        filename = self.make_filename("Items_keyw", output_dir)

        with open(filename, "w") as f:
            for item in topic_model.items_keyw.keys():
                f.write(str(item) + ' ' + str(topic_model.items_keyw[item]) + '\n')

    def save_propagation(self, propagation, step=0):
        '''
        Concrete method for saving the cascades files in a txt format
        '''
        output_dir = self.make_output_directory(self._path)
        if step == 0:
            filename = self.make_filename("Propagations", output_dir, True)
        else:
            filename = self.make_filename("Propagations", output_dir, False)

        with open(filename, 'a') as f:
            for node in range(len(propagation)):
                f.write(str(step) +' '+ str(propagation[node]))
