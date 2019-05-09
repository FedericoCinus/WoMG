# /diffusion/diffusion_model.py
# Abstract class defining Diffusion models
import abc
import pickle
import pathlib
from tqdm import tqdm


class DiffusionModel(abc.ABC):
    '''
    Abstract class for diffusion models

    Attributes
    ----------
    Hidden_network_model : networkModel obj
    Hidden_topic_model : topicModel obj

    Methods
    -------
    validate_config : concrete
        validates the correct tlt components
    run : concrete
        run the iteration method steps time
    iteration : abstract
        defines the one step iteration
    stop_criterior
        defines how to stop simulation
    save_model_attr
        saves all the attributes
    save_model_class
        saves all the class in pickle file

    Notes
    -----
    Hidden attributes names has to start with "_"
    '''

    def __init__(self):
        self.Hidden_network_model = {}
        self.Hidden_topic_model = {}


    def validate_config(self):
        '''
        Concrete method for validate the tlt components
        '''
        print('I am validating')

    def run(self, numb_steps):
        '''
        Simulates by running iteration method numb_steps times

        Parameters
        ----------
        numb_steps : int
            number of simulation steps, i.e. number of times to call iteration()
        '''
        print("Computing cascades: ")
        for t in tqdm(range(numb_steps)):
            if not self.stop_criterior()[0]:
                #print(self.stop_criterior()[0], self.stop_criterior()[1])
                self.iteration(step=t)
            if self.stop_criterior()[0] or self.stop_criterior_2():
                print('Simulation stoped at timestep ',str(t)   )


    @abc.abstractmethod
    def iteration(self, step):
        '''
        Abstract method for defining the iteration of the diffusion process:
        how nodes will activate on items in a fixed step
        '''
        pass

    @abc.abstractmethod
    def stop_criterior(self):
        '''
        Abstract method for defining a particular case in which simulation has
        to stop
        '''
        pass


    def save_model_attr(self, path=None, fformat='pickle', way='a', step=0):
        '''
        Saves all diffusion model attributes

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
         All the attributes which start with "_" are NOT saved
        '''
        if path == None or path == '':
            output_dir = pathlib.Path.cwd() / "Output"
        else:
            output_dir = str(path)

        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        for attribute in self.__dict__.keys():
            if not str(attribute).startswith('Hidden_'):
                # dealing with already existing files
                if str(attribute).startswith('_'):
                    attribute = str(attribute)[1:]
                filename = output_dir / str('Diffusion_' + str(attribute) + '_sim0' +'.' + str(fformat))
                sim_numb = 0
                while pathlib.Path(filename).exists():
                    sim_numb+=1
                    filename = output_dir / str('Diffusion_' + str(attribute) + '_sim' + str(sim_numb) +'.' + str(fformat))
                if step != 0:
                    sim_numb -= 1
                    filename = output_dir / str('Diffusion_' + str(attribute) + '_sim' + str(sim_numb) +'.' + str(fformat))

                title = 'time ' + str(step) + ': item; node'
                if fformat == 'txt':
                    with open(filename, way) as f:
                        f.write('\n'+title+'\n'+str(self.__getattribute__(str(attribute))))
                if fformat == 'pickle':
                    with open(filename, way+'b') as f:
                        pickle.dump('\n'+title+'\n'+self.__getattribute__(str(attribute)), f, pickle.HIGHEST_PROTOCOL)


    def save_model_class(self):
        '''
        Saves all class in pickle format in the current directory

        Notes
        -----
        Class model file will be saved with a name that starts with "Hidden_"
        '''
        file = pathlib.Path.cwd() /  '__diffusion_model'
        with open(file, 'wb') as f:
            pickle.dump(self, f)