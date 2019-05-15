# /topic/topic_model.py
# Abstract class defining Topic model
import abc
import pathlib
import pickle


class TopicModel(abc.ABC):
    '''
    Abstract class for topic models

    Attributes
    ----------
    topic_distrib : dict
        topic distribution i.e. probability distribution over numb_topics space

    Methods
    -------
    save_model_attr : concrete
        saves all the attributes
    save_model_class : concrete
        saves all the class in pickle file

    Notes
    -----
    Hidden attributes names has to start with "Hidden_"
    '''

    def __init__(self):
        self.topic_distrib = {}


    def save_model_attr(self, path=None, fformat='pickle', way='w'):
        '''
        Saves all topic model attributes

        Parameters
        ----------
        - path : string
          path in which the method will save the data,
          if None is given it will create an "Output" directory in the
          current path
        - fformat : string
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

                filename = output_dir / str("Topic_" + str(attribute) + '_sim0' +'.' + str(fformat))
                sim_numb = 1
                while pathlib.Path(filename).exists():
                    filename = output_dir / str('Topic_' + str(attribute) + '_sim'+str(sim_numb) +'.' + str(fformat))
                    sim_numb+=1

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
        file = pathlib.Path.cwd() / '__topic_model'
        with open(file, 'wb') as f:
            pickle.dump(self, f)
