# /Topic/lda.py
# Implementation of LDA topic-model
import pathlib
import gensim
import numpy as np
from topic.tlt_topic_model import TLTTopicModel
from utils.utility_functions import count_files, read_docs
from utils.distributions import random_viralities_vec

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

    def __init__(self, numb_topics, numb_docs, path_in):
        super().__init__()
        self._numb_topics = numb_topics
        self._numb_docs = numb_docs
        self._path_in = path_in
        self.items_keyw = {}


    def fit(self):
        '''
        Pipeline for fitting the lda model

        1. define the lda mode : generative mode / reading mode
        2. set the topic distributions of the documents
        3. set the word distribution of the documents
        '''
        mode_reading = self.set_lda_mode()
        lda_model = self.train_lda()
        self.topics_descript = self.get_topics_descript(lda_model)
        if mode_reading:
            self.get_items_descript(path=self._path_in, model=lda_model)
            self.get_items_keyw(model=lda_model)
        else:
            self.gen_items_descript(model=lda_model)
            self.get_items_keyw(model=lda_model)



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
        if ((self._numb_docs == None) and (self._path_in == None)):
            reading = False
            self._numb_docs = 100
        elif self._numb_docs:
            reading = False
        elif self._path_in:
            reading = True
        elif self._path_in and self._numb_docs:
            print('Error: insert docs for generating them; insert docs path for reading them. Not Both!')

        # setting lda input

        if reading:
            if self._path_in:
                self._input_path = pathlib.Path(self._path_in)
                numb_docs = count_files(self._input_path)
                if numb_docs != 0:
                    self._numb_docs = numb_docs
                    print('Extracting topic distribution from docs in ', self._input_path)
                else:
                    print('No txt file in: ', self._input_path)
            else:
                self._input_path = pathlib.Path.cwd().parent / "data" / "docs"
                if pathlib.Path(self._input_path).exists():
                    numb_docs = count_files(self._input_path)
                    if numb_docs != 0:
                        self._numb_docs = numb_docs
                        print('Extracting topic distribution from docs in ', self._input_path)
                    else:
                        print('No txt file in: ', self._input_path)
                else:
                    print('Docs folder inside input folder has been deleted')

        else:
            print('Setting LDA in generative mode: ', self._numb_docs, ' documents, with ', self._numb_topics, ' topics.')
            print('Training the LDA model ..')
            self._input_path = None
            self._training_path = pathlib.Path.cwd().parent / "data" / "docs" / "training_corpus2"

        return reading

    def set_docs_viralities(self, virality):
        '''
        Sets the documents viralities to the given scalar/vector

        Parameters
        ----------
        viralitiy : float
            Exponent of the powerlaw distribution for documents
            viralities. P(x; a) = x^{-a}, 0 <= x <=1
        '''
        viralities = random_viralities_vec(gamma=virality, dimensions=self._numb_docs)

        if np.size(viralities) == self._numb_docs:
            for item in range(self._numb_docs):
                self.viralities[item] = viralities[item]
        if np.size(viralities) == 1:
            for item in range(self._numb_docs):
                self.viralities[item] = viralities
        #print(viralities)


    def get_items_descript(self, path, model):
        '''
        Gets the topic distribution as attribute of the superclass
        '''
        #potrebbe esserci un bug
        docs = read_docs(path)
        id2word, corpus = self.preprocess_texts(docs)
        items_descript = model.get_document_topics(corpus, minimum_probability=0.)
        gammas = {}
        item = 0
        for item_descript in items_descript:
            gammas[item] = np.array([i[1] for i in item_descript])
            item += 1
        if self._numb_docs == len(gammas.keys()):
            print("Items' distribution over topics is stored")
            print(gammas)
        self.items_descript = gammas

    def gen_items_descript(self, model):
        '''
        Generates the topic distribution for each item
        and stores it in the imets_descript attribute
        '''
        alpha = model.alpha
        #alpha = [0.01120081, 0.04134526, 0.5296952,  0.00861911, 0.00862031, 0.01053169, 0.01223436, 0.1643439,  0.00871354, 0.00967268, 0.01102241, 0.01131404, 0.0118466,  0.02180933, 0.0123167]
        gammas = {}
        for item in range(self._numb_docs):
            gammas[item] = np.random.dirichlet(alpha)
        self.items_descript = gammas


    def get_items_keyw(self, model):
        '''
        Get the the items keyword in a bow format
        '''
        return


    def preprocess_texts(self, docs):
        '''
        Preprocessing input texts: divides docs into words, bow format
        '''
        data_words = self.sent_to_words(docs)
        dict_corp_tuple = self.corpus_gen(data_words)
        #print(dict_corp_tuple)
        return dict_corp_tuple


    def sent_to_words(self, docs):
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
        for sentence in docs:
            data_words.append(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        return list(data_words)

    def corpus_gen(self, data_words):
        '''
        Returns dictionary and corpus in the Bag of Words format

        Parameters
        ----------
        data_words : list
            list of lists: each entry is the list of words of one doc

        Returns
        -------
        id2word : list
            dictionary of unique words
        corpus : list
            list of lists of bag of words (id-count tuples)
        '''
        id2word = gensim.corpora.Dictionary(data_words)  # defining dictionary
        corpus = [id2word.doc2bow(word) for word in data_words]

        #print('Corpus with '+str(self._numb_docs)+' documents loaded')
        return id2word, corpus


    def train_lda(self):
        '''
        Pre-train lda model on a saved corpus for infering the prior weights of the distributions given
        a number of topics, which correpsonds to the weights' dimension

        Returns gensim object of alpha prior topic distrib vec

        Parameters
        ----------
        dict_corp_tuple : tuple
            output of the corpus_gen() method: dictionary and corpus

        Returns
        -------
        gensim lda alpha
        '''
        docs = read_docs(self._training_path)
        id2word, corpus = self.preprocess_texts(docs)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=self._numb_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
        return lda_model

    def get_topics_descript(self, model, mtrx_form=False):
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
        else:
            return model.print_topics()
