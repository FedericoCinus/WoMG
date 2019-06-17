# /Topic/lda.py
# Implementation of LDA topic-model
import re
import pathlib
import gensim
import numpy as np
from nltk.corpus import stopwords
from topic.tlt_topic_model import TLTTopicModel
from utils.utility_functions import count_files, read_docs, TopicsError
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

    def __init__(self, numb_topics, numb_docs, docs_path, items_descr_path):
        super().__init__()
        self.numb_topics = numb_topics
        self.numb_docs = numb_docs
        self.numb_words = 20
        self._docs_path = docs_path
        self._items_descr_path = items_descr_path
        self.items_keyw = {}


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
            self.items_descript, self.numb_docs = self.load_items_descr(self._items_descr_path)
        else:
            self.make_dictionary()
            lda_model = self.train_lda()
            if mode == 'get':
                self.topics_descript = self.get_topics_descript(lda_model)
                self.get_items_descript(path=self._docs_path, model=lda_model)
                self.get_items_keyw(path=self._docs_path, model=lda_model)
            if mode == 'gen':
                self.topics_descript = self.get_topics_descript(lda_model)
                self.gen_items_descript(model=lda_model)
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
        # setting mode
        if self.numb_docs == None and self._docs_path == None:
            mode = 'load'
            if self._items_descr_path == None:
                # pre-trained topic model with 15 topics and 50 docs
                self.items_descr_path = ' '
            else:
                # given trained topic model
                self.items_descript = self._items_descr_path

        if self.numb_docs == None and self._docs_path != None and self._items_descr_path == None:
            mode = 'get'
            path = pathlib.Path(self._docs_path)
            numb_docs = count_files(path)
            if numb_docs != 0:
                self.numb_docs = numb_docs
                print('Extracting topic distribution from docs in ', path)
            else:
                print('No txt file in: ', path)
            self._training_path = pathlib.Path.cwd().parent / "data" / "docs" / "training_corpus2"


        if self.numb_docs != None and self._docs_path == None and self._items_descr_path == None:
            mode = 'gen'
            print('Setting LDA in generative mode: ', self.numb_docs, ' documents, with ', self.numb_topics, ' topics.')
            print('Training the LDA model ..')
            self._training_path = pathlib.Path.cwd().parent / "data" / "docs" / "training_corpus2"

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
        viralities = random_viralities_vec(gamma=virality, dimensions=self.numb_docs)

        if np.size(viralities) == self.numb_docs:
            for item in range(self.numb_docs):
                self.viralities[item] = viralities[item]
        if np.size(viralities) == 1:
            for item in range(self.numb_docs):
                self.viralities[item] = viralities
        #print(viralities)


    def get_items_descript(self, path, model):
        '''
        Gets the topic distribution as attribute of the superclass
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
            #print(gammas)
        self.items_descript = gammas

    def gen_items_descript(self, model):
        '''
        Generates the topic distribution for each item
        and stores it in the imets_descript attribute
        '''
        alpha = model.alpha
        #alpha = [0.01120081, 0.04134526, 0.5296952,  0.00861911, 0.00862031, 0.01053169, 0.01223436, 0.1643439,  0.00871354, 0.00967268, 0.01102241, 0.01131404, 0.0118466,  0.02180933, 0.0123167]
        gammas = {}
        for item in range(self.numb_docs):
            gammas[item] = np.random.dirichlet(alpha)
        self.items_descript = gammas


    def get_items_keyw(self, path, model):
        '''
        Get the items keyword in a bow format
        '''
        return

    def gen_items_keyw(self, model):
        '''
        Generates the items keyword in a bow format
        '''
        items_descript = self.items_descript
        topics_descript = self.get_topics_descript(model, mtrx_form=True)

        for item in range(self.numb_docs):
            item_keyw = []
            for word in range(self.numb_words):
                multi_items = np.random.multinomial(1, items_descript[item], size=1)
                topic_index = np.where(multi_items==1)[1]
                multi_topics = np.random.multinomial(1, topics_descript[topic_index][0], size=1)
                word_index = np.where(multi_topics==1)[1]
                item_keyw.append(self.dictionary[word_index[0]])
            self.items_keyw[item] = self.to_bow(item_keyw)

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
        with open(self._items_descr_path, 'r') as f:
            numb_docs = 0
            for line in f:
                line = line.replace(',','').replace(']','').replace('[','')
                values = line.split()
                node = int(values[0])
                interests_vec = [float(i) for i in values[1:]]
                if self.numb_topics != len(interests_vec):
                    raise TopicsError("Please write the correct number of topics as input")
                items_descr_dict[node] = interests_vec
                numb_docs += 1
        return items_descr_dict, numb_docs


    def to_bow(self, text):
        '''
        Returns the give doc in a bow format
        '''
        bow = self.dictionary.doc2bow(text)
        bow_dict = {}
        for word_index, freq in bow:
            bow_dict[self.dictionary[word_index]] = freq
        return bow_dict

    @staticmethod
    def clean_text(docs):
        '''

        Parameters
        ----------
        docs : list of lists
            each entry is the list containing the doc in string format

        '''
        print(docs)
        # Remove new line characters
        data = [re.sub(r'\s+', ' ', str(sent[0])) for sent in docs]

        # Remove distracting single quotes
        data = [re.sub(r"\'", "", str(sent[0])) for sent in docs]

        # Remove all the special characters
        data = [re.sub(r'\W', ' ', str(sent[0])) for sent in docs]

        # remove all single characters
        data = [re.sub(r'\s+[a-zA-Z]\s+', ' ', str(sent[0])) for sent in docs]

        # Remove single characters from the start
        data = [re.sub(r'\^[a-zA-Z]\s+', ' ', str(sent[0])) for sent in docs]

        # Substituting multiple spaces with single space
        data = [re.sub(r'\s+', ' ', str(sent[0]), flags=re.I) for sent in docs]

        # Removing prefixed 'b'
        data = [re.sub(r'^b\s+', '', str(sent[0])) for sent in docs]

        # Remove article
        data = [re.sub(r'the', '', str(sent[0])) for sent in docs]

        # Remove to
        data = [re.sub(r'to', '', str(sent[0])) for sent in docs]

        #print(data)
        return data

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
        for sentence in docs:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


    def remove_stop_words(self, texts):
        '''
        '''
        stop_words = stopwords.words('english')
        return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]



    def make_dictionary(self):
        '''
        Sets the dictionary attribute using a training corpus in input

        Parameters
        ----------
        data_words : list
            list of lists: each entry is the list of words of one doc

        Returns
        -------
        id2word : list
            dictionary of unique words
        '''
        train_corpus = read_docs(self._training_path)
        clean_train_corpus = self.clean_text(train_corpus)
        dictio = list(self.sent_to_words(clean_train_corpus))
        self.dictionary = gensim.corpora.Dictionary(dictio)


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
        docs = read_docs(self._training_path)
        #clean_docs = self.clean_text(docs)
        docs_words = list(self.sent_to_words(docs))
        #print(docs_words)
        corpus = [self.dictionary.doc2bow(text, return_missing=True) for text in docs_words]
        #print(corpus)
        #print(self.dictionary)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=self.dictionary,
                                                    num_topics=self.numb_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

        return lda_model
