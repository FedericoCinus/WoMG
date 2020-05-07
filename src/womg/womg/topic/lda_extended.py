'''
/Topic/lda.py
Implementation of LDA topic-model
'''
import re
import os
import pathlib
import gensim
import numpy as np
from womg_core.topic.lda import LDA
from womg_core.utils.utility_functions import count_files, read_docs
from womg_core.utils.distributions import random_powerlaw_vec

class LDAExtended(LDA):
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
                 docs_path,
                 items_descr,
                 progress_bar):
        super().__init__(numb_topics,
                         numb_docs,
                         items_descr,
                         progress_bar)
        self._docs_path = docs_path




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
            super().fit()

        if mode == 'get':
            lda_model = self.train_lda()
            self.topics_descript = self.get_topics_descript(lda_model)
            self.get_items_descript(path=self._docs_path, model=lda_model)
            self.get_items_keyw(path=self._docs_path)
        if mode == 'gen':
            lda_model = self.train_lda()
            self.topics_descript = self.get_topics_descript(lda_model)
            super().gen_items_descript()
            self.gen_items_keyw(model=lda_model)



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
        if self.numb_docs is None and self._docs_path is not None and self._items_descr is None:
            mode = 'get'
            path = pathlib.Path(self._docs_path)
            numb_docs = count_files(path)
            if numb_docs != 0:
                self.numb_docs = numb_docs
            else:
                print('No txt file in: ', path)
        elif self.numb_docs is not None and self._docs_path is not None and self._items_descr is None:
            mode = 'get'
        elif self.numb_docs is not None and self._docs_path is None and self._items_descr is None:
            mode = 'gen'
        else:
            mode = super().set_lda_mode()

        return mode


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
        print('Extracting topic distribution from docs in ', path)
        docs = read_docs(path, verbose=False)
        corpus = self.preprocess_texts(docs)
        gammas = {}
        item = 0
        for item in range(self.numb_docs):
            item_descript = model.get_document_topics(corpus[item], minimum_probability=0.)
            gammas[item] = np.array([i[1] for i in item_descript])
            item += 1
        if self.numb_docs == len(gammas.keys()):
            pass
        self.items_descript = gammas



    def get_items_keyw(self, path):
        '''
        Get the items keyword in a bow format
        '''
        docs = read_docs(path)
        preproc_docs = [gensim.parsing.preprocessing.remove_stopwords((sent[0].lower())) for sent in docs if len(sent) != 0]
        data_words = self.sent_to_words(preproc_docs, verbose=False)
        for item in range(self.numb_docs):
            self.items_keyw[item] = self.to_bow(data_words[item])


    def gen_items_keyw(self, model):
        '''
        Generates the items keyword in a bow format
        '''
        items_descript = self.items_descript
        topics_descript = self.get_topics_descript(model, mtrx_form=True)

        for item in range(self.numb_docs):
            item_keyw = []
            for _ in range(self.numb_words):
                multi_items = np.random.multinomial(1, items_descript[item], size=1)
                topic_index = np.where(multi_items == 1)[1]
                multi_topics = np.random.multinomial(1, topics_descript[topic_index][0], size=1)
                word_index = np.where(multi_topics == 1)[1]
                item_keyw.append(self.dictionary[word_index[0]])
            self.items_keyw[item] = self.to_bow(item_keyw)

    @staticmethod
    def get_topics_descript(model, mtrx_form=False):
        '''
        Getting the word distribution for each topic

        Parameters
        ----------
        model : gensim obj
            input model from gensim library e.g. lda model trained
        all : bool
            if True the complete word distribution for each topic is given
            in a (topics)x(words) matrix format
            if False a combination of most important words is given for each topic
        '''
        if mtrx_form:
            return model.get_topics()
        return model.print_topics()


    def to_bow(self, text):
        '''
        Returns the give doc in a bow format
        '''
        bow = self.dictionary.doc2bow(text)
        bow_dict = {}
        for word_index, freq in bow:
            bow_dict[self.dictionary[word_index]] = freq
        return bow_dict

    def preprocess_texts(self, docs):
        '''
        Preprocessing input texts: divides docs into words, bow format
        '''
        data = [gensim.parsing.preprocessing.remove_stopwords((sent[0].lower())) for sent in docs if len(sent) != 0]

        # Remove new line characters
        data = [re.sub(r'\s+', ' ', str(sent)) for sent in data]

        # Remove distracting single quotes
        data = [re.sub(r"\'", "", str(sent)) for sent in data]

        # Remove all the special characters
        data = [re.sub(r'\W', ' ', str(sent)) for sent in data]

        # remove all single characters
        data = [re.sub(r'\s+[a-zA-Z]\s+', ' ', str(sent)) for sent in data]

        # Remove single characters from the start
        data = [re.sub(r'\^[a-zA-Z]\s+', ' ', str(sent)) for sent in data]

        # Substituting multiple spaces with single space
        data = [re.sub(r'\s+', ' ', str(sent), flags=re.I) for sent in data]

        # Removing prefixed 'b'
        data = [re.sub(r'^b\s+', '', str(sent)) for sent in data]

        # Remove article
        data = [re.sub(r'the', '', str(sent)) for sent in data]

        # Remove to
        data = [re.sub(r'to', '', str(sent)) for sent in data]

        data_words = self.sent_to_words(data, verbose=False)
        corpus = [self.dictionary.doc2bow(word) for word in data_words]
        return corpus


    def sent_to_words(self, docs, verbose=True):
        '''
        Returns list of docs divided in words

        Parameters
        ----------
        docs : list
            list of lists: each entry is the list containing the document as a string

        Returns
        -------
        data_words : list
            list of lists: each entry is the list of words of one doc
        '''
        data_words = []
        if verbose:
            for sentence in self._progress_bar(docs):
                data_words.append(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        else:
            for sentence in docs:
                data_words.append(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        return list(data_words)


    def train_lda(self):
        '''
        Pre-train lda model on a saved corpus for infering the prior weights of
        the distributions given a number of topics, which correpsonds to the
        weights' dimension

        Returns gensim object of lda model

        Parameters
        ----------
        path : str
            path of the training corpus

        Returns
        -------
        gensim lda model object
        '''
        print('Training LDA model..')
        docs = read_docs(self._training_path)
        data_words = self.sent_to_words(docs, verbose=False)

        self.dictionary = gensim.corpora.Dictionary(data_words)
        corpus = self.preprocess_texts(docs)

        hash_name = 'trained_lda'+str(hash(str(self._training_path)+str(self.numb_topics)))+'.model'
        saved_model_fname = self.main_data_path/'topic_model'/hash_name
        if os.path.exists(os.path.abspath(saved_model_fname)):
            lda_model = gensim.models.ldamodel.LdaModel.load(os.path.abspath(saved_model_fname))
        else:
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                        id2word=self.dictionary,
                                                        num_topics=self.numb_topics,
                                                        random_state=100,
                                                        update_every=1,
                                                        chunksize=100,
                                                        passes=10,
                                                        alpha='auto',
                                                        per_word_topics=True)

            lda_model.save(os.path.abspath(saved_model_fname))

        return lda_model
