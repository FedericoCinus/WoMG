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
    items_descript : dict
        topic distribution for each item over the topics space
        key: item index, value: numb_topics dim vector
    topics_descript : dict
        word distribution for each topic over thw words space
        key: topic index, value: numb_words dim vector

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
        self.items_descript = {}
        self.topics_descript = {}
