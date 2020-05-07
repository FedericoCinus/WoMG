'''
Class containing all womg outputs 
'''
import numpy as np

class Propagation:
    def __init__(self, network_model, topic_model, diffusion_model):
        self.diffusion_model = diffusion_model
        self.topic_model = topic_model
        self.network_model = network_model

        self._propagations = None
        self._docs = None
        self._topic_distributions = None
        self._interests = None
        self._topics_descriptions = None


    @property
    def propagations(self):
        '''
        list of M tuples (time, node)
        '''
        self.propagations = self.diffusion_model.all_propagations
        return self._propagations

    @propagations.setter
    def propagations(self, value):
        self._propagations = value

    @property
    def docs(self):
        print(self.topic_model.items_keyw)
        self.docs = [v for k, v in self.topic_model.items_keyw.items()]
        return self._docs

    @docs.setter
    def docs(self, value):
        self._docs = value


    @property
    def topic_distributions(self):
        '''
        KxM matrix
        '''
        print(self.topic_model.items_descript)
        vectors = [v for v in self.topic_model.items_descript.values()]
        self.topic_distributions = np.column_stack(vectors)
        return self._topic_distributions

    @topic_distributions.setter
    def topic_distributions(self, value):
        self._topic_distributions = value

    @property
    def interests(self):
        '''
        kxN matrix
        '''
        interests = [v for v in self.network_model.users_interests.values()]
        self.interests = np.column_stack(interests)
        return self._interests

    @interests.setter
    def interests(self, value):
        self._interests = value


    @property
    def topics_descriptions(self):
        '''
        list of linear combination of words
        '''
        self._topics_descriptions = self.topic_model.topics_descript
        return self._topics_descriptions

    @topics_descriptions.setter
    def topics_descriptions(self, value):
        self._topics_descriptions = value
