'''/Topic/lda.py
Implementation of LDA topic-model
'''
import os
import pathlib
import numpy as np
from tqdm import tqdm, tqdm_notebook
import womg_core
from womg_core.topic.tlt_topic_model import TLTTopicModel
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
    def __init__(self, numb_topics,
                 numb_docs,
                 items_descr,
                 progress_bar):
        super().__init__()
        self.numb_topics = numb_topics
        self.numb_docs = numb_docs
        self.numb_words = 20
        self._docs_path = None
        self._items_descr = items_descr
        self.items_keyw = {}
        self.dictionary = []
        self.main_data_path = pathlib.Path(os.path.abspath(womg_core.__file__).replace('/womg_core/__init__.py', ''))/'womgdata'
        self._training_path = self.main_data_path /'docs'/'training_corpus_ap'
        self._topics_descr_path = self.main_data_path / 'topic_model' / 'Topics_descript.txt'
        self.topics_descript = self.load_topics_descr(self._topics_descr_path)
        if progress_bar:
            self._progress_bar = tqdm_notebook
        else:
            self._progress_bar = tqdm


    def fit(self):
        '''
        Pipeline for fitting the lda model

        1. define the lda mode : generative mode / reading mode
        2. train lda
        3. get the items descriptions (topic distribution for each item)
        4. get the items keywords (bow list for each item)
        '''
        mode = self.set_lda_mode()
        if mode == 'load':
            if isinstance(self._items_descr, dict):
                self.items_descript = self._items_descr
                self.numb_docs = len(self._items_descr.keys())
            else:
                self.items_descript, self.numb_docs = self.load_items_descr(self._items_descr)
        if mode == 'gen':
            self.gen_items_descript()

    @staticmethod
    def load_topics_descr(path):
        '''
        Loads a topic description file (for each topic word distribution)
        '''
        with open(path, 'r') as file:
            topics_descript = file.readlines()
        return str(topics_descript[0])

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
        # setting mode
        if self.numb_docs is None and self._docs_path is None:
            mode = 'load'
            if self._items_descr is None:
                # pre-trained topic model with 15 topics and 50 docs
                self._items_descr = self.main_data_path / 'topic_model' / 'Items_descript.txt'
            else:
                pass
            if isinstance(self._items_descr, dict):
                print('Loading items descriptions (topic distrib for each doc)')
            else:
                print('Loading items descriptions (topic distrib for each doc) in: ',
                      self._items_descr)
        if self.numb_docs is not None and self._docs_path is None and isinstance(self._items_descr, dict):
            mode = 'load' # when womg extended has been executed in gen mode

        elif self.numb_docs is None and self._docs_path is not None and self._items_descr is None:
            print('Please install the womg extended version')

        elif self.numb_docs is not None and self._docs_path is not None and self._items_descr is None:
            print('Please install the womg extended version')

        elif self.numb_docs is not None and self._docs_path is None and self._items_descr is None:
            mode = 'gen'
            print('Setting LDA in generative mode: ',
                  self.numb_docs, ' documents, with ',
                  self.numb_topics, ' topics.')
            print('Training the LDA model ..')

        return mode


    def set_docs_viralities(self, virality_exp):
        '''
        Sets the documents viralities to the given scalar/vector

        Parameters
        ----------
        viralitiy : float
            Exponent of the pareto distribution for documents
            viralities.
        '''
        viralities = random_powerlaw_vec(gamma=virality_exp, dimensions=self.numb_docs)

        if np.size(viralities) == self.numb_docs:
            for item in range(self.numb_docs):
                self.viralities[item] = viralities[item]
        if np.size(viralities) == 1:
            for item in range(self.numb_docs):
                self.viralities[item] = viralities[0]

    def gen_items_descript(self):
        '''
        Generates the topic distribution for each item
        and stores it in the items_descript attribute
        '''
        print('generating items descript')
        alpha = [1.0 / self.numb_topics for i in range(self.numb_topics)]
        gammas = {}
        for item in range(self.numb_docs):
            gammas[item] = np.random.dirichlet(alpha)
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
        with open(path, 'r') as file:
            numb_docs = 0
            for line in file:
                line = line.replace(',', '').replace(']', '').replace('[', '')
                values = line.split()
                node = int(values[0])
                interests_vec = [float(i) for i in values[1:]]
                if self.numb_topics != len(interests_vec):
                    print("Please write the correct number of topics",
                          "as input or in case you give the items_descr_path",
                          " you can omit it")
                items_descr_dict[node] = interests_vec
                numb_docs += 1
        return items_descr_dict, numb_docs
