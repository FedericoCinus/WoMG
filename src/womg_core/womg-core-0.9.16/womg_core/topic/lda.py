# /Topic/lda.py
# Implementation of LDA topic-model
import os
import womg_core
import pathlib
import numpy as np
from womg_core.topic.tlt_topic_model import TLTTopicModel
from womg_core.utils.utility_functions import read_docs, TopicsError
from womg_core.utils.distributions import random_powerlaw_vec

class LDA(TLTTopicModel):
    '''
    Class implementing Latent Dirichlet Allocation as topic model
    for topic distribution involved in tlt class model

    Attributes
    ----------
    numb_topics : int
        hidden
    items_keyw : dict
        dict of items in bow format

    Methods
    -------
    set_lda_mode : concrete
        sets the lda mode: reading or generating mode
    '''

    def __init__(self, numb_topics, numb_docs, docs_path, items_descr_path):
        super().__init__()
        self.numb_topics = numb_topics
        self.numb_docs = numb_docs
        self._docs_path = docs_path
        self._items_descr_path = items_descr_path
        self._training_path = pathlib.Path.cwd().parent / "womgdata" / "docs" / "training_corpus2"


    def fit(self):
        '''
        Pipeline for fitting the lda model

        1. define the lda mode : generative mode / reading mode
        2. train lda
        3. get the items descriptions (topic distribution for each item)
        4. get the items keywords (bow list for each item)
        '''
        print('\n In fit method there are ', self.numb_docs, self._docs_path, self._items_descr_path)
        mode = self.set_lda_mode()
        if mode == 'load':
            self.items_descript, self.numb_docs = self.load_items_descr(self._items_descr_path)
        if mode == 'gen':
            self.gen_items_descript()



    def set_lda_mode(self):
        '''
        Sets how lda has to work:
        reading a document folder or generating documents


        Parameters
        ----------
        path : string
            position of the document folder

        Returns
        -------
        reading : bool
            if True: it will read docs inside the given folder path or input folder
            if False: it will use lda for generating docs

        '''
        print(self.numb_docs, self._docs_path, self._items_descr_path, flush=True)
        # setting mode
        if self.numb_docs == None and self._docs_path == None:
            mode = 'load'
            if self._items_descr_path == None:
                # pre-trained topic model with 15 topics and 50 docs
                self._items_descr_path = pathlib.Path(os.path.abspath(womg_core.__file__)[:-21])  / 'womgdata' / 'topic_model' / 'Items_descript.txt'
                self._topics_descr_path = pathlib.Path(os.path.abspath(womg_core.__file__)[:-21])  / 'womgdata' / 'topic_model' / 'Topics_descript.txt'
                self.topics_descript = self.load_topics_descr(self._topics_descr_path)

            print('Loading items descriptions (topic distrib for each doc) in: ', self._items_descr_path)

        elif self.numb_docs != None and self._docs_path == None and self._items_descr_path == None:
            mode = 'gen'
            #print('Setting LDA in generative mode: ', self.numb_docs, ' documents, with ', self.numb_topics, ' topics.')

        else:
            print('Error: ', 'number of docs ', self.numb_docs, ', docs path ', self._docs_path, ', items descriptions ', self._items_descr_path)

        return mode


    def set_docs_viralities(self, virality):
        '''
        Sets the documents viralities to the given scalar/vector

        Parameters
        ----------
        viralitiy : float
            Exponent of the powerlaw distribution for documents
            viralities. P(x; a) = x^{-a}, 0 <= x <=1
        '''
        viralities = random_powerlaw_vec(gamma=virality, dimensions=self.numb_docs)

        if np.size(viralities) == self.numb_docs:
            for item in range(self.numb_docs):
                self.viralities[item] = viralities[item]
        if np.size(viralities) == 1:
            for item in range(self.numb_docs):
                self.viralities[item] = viralities[0]
        #print(viralities)

    def gen_items_descript(self):
        '''
        Generates the topic distribution for each item
        and stores it in the items_descript attribute
        '''
        #print('Genereting items descriptions')
        alpha =  [1.0 / self.numb_topics for i in range(self.numb_topics)]
        gammas = {}
        for item in range(self.numb_docs):
            gammas[item] = np.random.dirichlet(alpha)
        self.items_descript = gammas

    def get_items_descript(self, path, model):
        '''
        Gets the topic distribution and puts it as attribute of the superclass
        reading documents from the docs folder path given as parameter

        Parameters
        ----------
        path : str
            path of the of the docs folder
        model : Gensim obj
            trained Gensim lda model
        '''
        docs = read_docs(path)
        corpus = self.preprocess_texts(docs)
        gammas = {}
        item = 0
        for item in range(self.numb_docs):
            item_descript = model.get_document_topics(corpus[item], minimum_probability=0.)
            gammas[item] = np.array([i[1] for i in item_descript])
            item += 1
        if self.numb_docs == len(gammas.keys()):
            print("Items' distribution over topics is stored")
        self.items_descript = gammas


    def load_items_descr(self, path):
        '''
        Returns the items_descript loaded from a file in path

        Parameters
        ----------
        path : int
            path of the items_descript file

        Returns
        -------
        tuple of items_descript loaded from path and numb_docs
        '''
        items_descr_dict = {}
        with open(path, 'r') as f:
            numb_docs = 0
            for line in f:
                line = line.replace(',','').replace(']','').replace('[','')
                values = line.split()
                node = int(values[0])
                interests_vec = [float(i) for i in values[1:]]
                if self.numb_topics != len(interests_vec):
                    raise TopicsError("Please write the correct number of topics as input or in case you give the items_descr_path you can omit it")
                items_descr_dict[node] = interests_vec
                numb_docs += 1
        return items_descr_dict, numb_docs

    def load_topics_descr(self, path):
        '''
        Loads a topic description file (for each topic word distribution)
        '''
        with open(path, 'r') as f:
            topics_descript = f.readlines()
            #topics_descript[0].replace("\\",'').replace(']','')
        return str(topics_descript[0])
