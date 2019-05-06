# /Topic/lda.py
# Implementation of LDA topic-model
import pathlib
import gensim
import random
import numpy as np
from topic.tlt_topic_model import TLTTopicModel
from utilities.utility_functions import count_files, read_docs
from utilities.distributions import random_viralities_vec

class LDA(TLTTopicModel):
    '''
    Class implementing Latent Dirichlet Allocation as topic model
    for topic distribution involved in tlt class model

    Attributes
    ----------
    numb_topics : int
        hidden
    word_distrib : dict
        key: topic index, value: word distribution

    Methods
    -------
    set_lda_mode : concrete
        sets the lda mode: reading or generating mode
    '''

    def __init__(self, numb_topics, numb_docs, virality, path_in):
        super().__init__()
        self.Hidden_numb_topics = numb_topics
        self.Hidden_numb_docs = numb_docs
        self.Hidden_word_distrib = {}
        self.set_lda_mode(path=path_in)
        self.set_viralities(virality)
        self.set_topic_distrib()
        self.set_word_distrib()
        self.save_model_class()

    def set_lda_mode(self, path, mode='r'):
        '''
        Sets how lda has to work:
        reading a document folder or generating documents


        Parameters
        ----------
        path : string
            position of the document folder
        mode : string bool
            if 'r': it will read docs inside the given folder path or input folder
            if 'g': it will use lda for generating docs

        '''
        # setting mode
        if ((self.Hidden_numb_docs == None) and (path == None)):
            mode = 'g'
            self.Hidden_numb_docs = 100
        elif self.Hidden_numb_docs:
            mode = 'g'
        elif path:
            mode = 'r'
        elif path and self.Hidden_numb_docs:
            print('Error: insert docs for generating them; insert docs path for reading them. Not Both!')

        # setting lda input
        if mode == 'g':
            print('Setting LDA in generative mode: K is set automatically to 15')
            self.Hidden_input_path = None
            self.Hidden_training_path = pathlib.Path.cwd() / "Input" / "Docs" / "text"
        if mode == 'r':
            if path:
                self.Hidden_input_path = pathlib.Path(path)
                numb_docs = count_files(self.Hidden_input_path)
                if numb_docs != 0:
                    self.Hidden_numb_docs = numb_docs
                    print('Extracting topic distribution from docs in ', self.Hidden_input_path)
                else:
                    print('No txt file in: ', self.Hidden_input_path)
            else:
                self.Hidden_input_path = pathlib.Path.cwd() / "Input" / "Docs"
                if pathlib.Path(self.Hidden_input_path).exists():
                    numb_docs = count_files(self.Hidden_input_path)
                    if numb_docs != 0:
                        self.Hidden_numb_docs = numb_docs
                        print('Extracting topic distribution from docs in ', self.Hidden_input_path)
                    else:
                        print('No txt file in: ', self.Hidden_input_path)
                else:
                    print('Docs folder inside input folder has been deleted')


    def set_viralities(self, virality):
        '''
        Sets the documents viralities to the given scalar/vector

        Parameters
        ----------
        viralities : int/vec
            if int is given: all the viralities are set to the given scalar
            if numb_docs vec is given: each doc virality is set to the correspondent
        '''
        viralities = random_viralities_vec(gamma=virality, dimensions=self.Hidden_numb_docs)

        if np.size(viralities) == self.Hidden_numb_docs:
            for item in range(self.Hidden_numb_docs):
                self.viralities[item] = viralities[item]
        if np.size(viralities) == 1:
            for item in range(self.Hidden_numb_docs):
                self.viralities[item] = viralities
        #print(viralities)


    def set_topic_distrib(self):
        '''
        Sets the topic distribution as attribute of the superclass

        1. read the docs
        2. creates the bag of words
        3. creates the corpus
        4. extracts the topic distribution for each doc
        5. formats the topic model of gensim to python dict
        '''
        if self.Hidden_input_path != None:
            docs = read_docs(self.Hidden_input_path)
            dict_corp_tuple = self.preprocess_texts(docs)
            topic_model = self.topic_distrib_extraction(dict_corp_tuple)
            gammas = self.tformat(topic_model)
            self.topic_distrib = gammas
        else:
            docs = read_docs(self.Hidden_training_path)
            dict_corp_tuple = self.preprocess_texts(docs)
            alpha = self.train_lda(dict_corp_tuple)
            self.topic_distrib = self.generates_topic_distrib(alpha)



    def preprocess_texts(self, docs):
        ''' 
        Preprocessing input texts: divides docs into words, bow format
        '''
        data_words = self.sent_to_words(docs)
        dict_corp_tuple = self.corpus_gen(data_words)

        #print(dict_corp_tuple)
        return dict_corp_tuple


    def set_word_distrib(self):
        '''
        Sets word distribution of topics as attribute
        '''
        pass

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
    
        #print('Corpus with '+str(self.Hidden_numb_docs)+' documents loaded')
        return id2word, corpus

    def topic_distrib_extraction(self, dict_corp_tuple):
        '''
        Returns gensim object of items' distribution over topics

        Parameters
        ----------
        dict_corp_tuple : tuple
            output of the corpus_gen() method: dictionary and corpus

        Returns
        -------
        gensim lda topic model
        '''
        id2word = dict_corp_tuple[0]
        corpus = dict_corp_tuple[1]
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=self.Hidden_numb_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)
        #print(lda_model.alpha)
        return lda_model.get_document_topics(corpus, minimum_probability=0.)

    def tformat(self, topic_model):
        '''
        Returns items' distribution over topics in dict format, each key is
        the item index

        Parameters
        ----------
        topic model : gensim obj
            lda topic model in gensim format

        Returns
        -------
        gammas : dict
            dictionary of topic distributions for documents
        '''
        gammas = {}
        item = 0
        for item_distr in topic_model:
            gammas[item] = np.array([i[1] for i in item_distr])
            item += 1
        if self.Hidden_numb_docs == len(gammas.keys()):
            print("Items' distribution over topics is stored")
            #print(gammas)
        return gammas

    def train_lda(self, dict_corp_tuple):
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
        id2word = dict_corp_tuple[0]
        corpus = dict_corp_tuple[1]
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=self.Hidden_numb_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

        return lda_model.alpha
        



    def generates_topic_distrib(self, alpha):
        '''
        Generates topic distribution for each hypothetical document using
        pre-trained lda over 15 topics
        '''
        #alpha = [0.01120081, 0.04134526, 0.5296952,  0.00861911, 0.00862031, 0.01053169, 0.01223436, 0.1643439,  0.00871354, 0.00967268, 0.01102241, 0.01131404, 0.0118466,  0.02180933, 0.0123167]
        gammas = {}
        for item in range(self.Hidden_numb_docs):
            gammas[item] = np.random.dirichlet(alpha)
        return gammas
