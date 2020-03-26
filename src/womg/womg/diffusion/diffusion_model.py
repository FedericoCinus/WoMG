# /diffusion/diffusion_model.py
# Abstract class defining Diffusion models
import abc
import pickle
import pathlib


class DiffusionModel(abc.ABC):
    '''
    Abstract class for diffusion models

    Attributes
    ----------
    network_model : networkModel obj
    topic_model : topicModel obj

    Methods
    -------
    run : concrete
        run the iteration method steps time
    iteration : abstract
        defines the one step iteration
    stop_criterior : abstract
        defines how to stop simulation
    '''

    def __init__(self):
        self.network_model = {}
        self.topic_model = {}

    def run(self, numb_steps):
        '''
        Simulates by running iteration method numb_steps times

        Parameters
        ----------
        numb_steps : int
            number of simulation steps, i.e. number of times to call iteration()
        '''
        print("Computing cascades: ")
        for t in self._progress_bar(range(numb_steps)):
            if t==0:
                self.iteration(step=0)
            else:
                if not self.stop_criterior():
                    self.iteration(step=t+1)
                else:
                    #print('\n stop_criterior: ', self.stop_criterior(), '\n')
                    print('\n Simulation stopped at timestep ', str(t-1) ,
                            '\n Diffusion has been completed.'  )
                    break


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
