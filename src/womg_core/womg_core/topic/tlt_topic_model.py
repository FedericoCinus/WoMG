'''/Topic/TopicModel.py
Abstract class defining Topic model
'''
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
    fit : abstract
        inferes/generates topic distribution for each document

    set_docs_viralities : abstract
        sets the documents viralities
    '''

    def __init__(self):
        super().__init__()
        self.viralities = {}

    @abc.abstractmethod
    def fit(self):
        '''Methods for infering/generating topic distributions of the given documents
        '''
        pass

    @abc.abstractmethod
    def set_docs_viralities(self, virality_exp):
        '''Method for setting the documents viralities
        '''
