# /Topic/TopicModel.py
# Abstract class defining Topic model
import abc
from womg.topic.topic_model import TopicModel


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
