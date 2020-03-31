class Propagation:
    def __init__(self, network_model, topic_model, diffusion_model):
        self.diffusion_model = diffusion_model
        self.topic_model = topic_model
        self.network_model = network_model

        self._propagations = None
        self._docs = None
        self._topic_distributions = None
        self._interests = None


    @property
    def propagations(self):
        self.propagations = self.diffusion_model.all_propagations
        return self._propagations

    @propagations.setter
    def propagations(self, value):
        self._propagations = value

    @property
    def docs(self):
        self.docs = self.topic_model.items_keyw
        return self._docs

    @docs.setter
    def docs(self, value):
        self._docs = value


    @property
    def topic_distributions(self):
        '''
        When propagations is called for save it is set to the current config
        '''
        self.topic_distributions = self.topic_model.items_descript
        return self._topic_distributions

    @topic_distributions.setter
    def topic_distributions(self, value):
        self._topic_distributions = value

    @property
    def interests(self):
        '''
        When propagations is called for save it is set to the current config
        '''
        self.interests = self.network_model.users_interests
        return self._interests

    @interests.setter
    def interests(self, value):
        self._interests = value
