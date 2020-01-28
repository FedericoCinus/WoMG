# /Topic/TopicModel.py
# Abstract class defining Topic model
import abc
from womg_core.topic.topic_model import TopicModel


class TLTTopicModel(TopicModel):
    '''
    Abstract class for topic models involved in tlt diffusion model class

    Attributes
    ----------
    viralities : dict
        key: document index, value: int virality parameter

    Methods
    -------
    topic_distrib_extraction : absract
        inferes topic distribution for each document
    validate_topic_config : concrete
        validate the configuration of the class model for the tlt class
    '''

    def __init__(self):
        super().__init__()
        self.viralities = {}

    @abc.abstractmethod
    def get_items_descript(self):
        '''
        Methods for infering topic distributions of the given documents
        '''
        pass

    def validate_topic_config(self):
        '''
        Checks the topic model structure for the tlt model

        topic_distrib attribute must be a dictionary in which:
        - key : int
          index of the item
        - value : array
          numb_topics dim array that is the topic distribution of the key-item
        '''
        pass
