#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
NATURAL LANGUAGE PROCESSING UTILS
Grouping various scripts and functions for nlp 


Started on the 16/01/2017

Word2vec : 
- In all languages - https://github.com/Kyubyong/wordvectors


theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""

import warnings
warnings.filterwarnings("ignore")


# Usual
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import json
from tqdm import tqdm
import operator
import string

# Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances


# Ekimetrics
from ekimetrics.nlp import utils as nlp_utils
from ekimetrics.network import utils as network_utils
from ekimetrics.utils.time import Timer

# Gensim
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.summarization import summarize,keywords




# Plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go



# Textblob
from textblob import TextBlob


# PyMongo
try:
    from pymongo import MongoClient
except:
    print("Skipped MongoDB pymongo wrapper")


# NLTK
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer # nltk v3.2.4+
except:
    pass


# SpaCy
try:
    import spacy
except:
    print("Skipped Spacy import")


# PyDeepL
try:
    import pydeepl
except:
    pass


#=============================================================================================================================
# FUNCTIONS USED IN CLASSES
#=============================================================================================================================



def isPunct(word):
    return len(word) == 1 and word in string.punctuation





def isNumeric(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except:
        return False


#=============================================================================================================================
# VARIABLES
#=============================================================================================================================


LANGUAGES_BLOBDEEPL_NLTK = {
    "en":"english",
    "fr":"french",
    "de":"german",
    "es":"spanish",
    "it":"italian",
    "pt":"portuguese",
    "it":"italian",
    "nl":"dutch",
    "pl":"polish",
}

LANGUAGES_NLTK_BLOBDEEPL = {v:k for k,v in LANGUAGES_BLOBDEEPL_NLTK.items()}





#=============================================================================================================================
# DOCUMENT CLASS
#=============================================================================================================================



class Document(object):
    """
    NLP base class to analyze a text 
    """
    def __init__(self,text = None,title = "",tokens = None,default_language = None,vector = None,polarity = 0.0,with_spacy = False):

        # Initialize parameters
        self.title = title
        self.vector = vector
        self.with_spacy = with_spacy
        self.cluster = None
        self.set_sentiment(polarity)

        # Load texts or tokens
        self.load_text_or_tokens(text = text,tokens = tokens)

        # Set the language of the document or detect it if not specified       
        self.set_language(language = default_language)



    def __repr__(self):
        """
        String representation of a document
        Returns the title or a text extract
        """
        if len(self.title) > 1:
            return self.title
        else:
            return self.text[:150]



    def __str__(self):
        """
        String representation of a document
        Returns the title or a text extract
        """

        return self.__repr__()


    def __len__(self):
        return len(self.text)





    def use_spacy(self):
        """
        Force the use of the SpaCy library
        """
        self.with_spacy = True




    def load_text_or_tokens(self,text = None,tokens = None):
        # Load the texts or the tokens
        if text is None:
            self.set_tokens(tokens)
        elif tokens is None:
            self.set_text(text)
        else:
            raise ValueError("You have to provide dataset")




    #--------------------------------------------------------------------------------------------------------
    # GETTERS

    def get_unique_tokens(self):
        """
        Gets the unique tokens of the text of a document
        :returns: list of str -- the list of unique tokens composing a document
        """
        return list(set(self.tokens))


    def get_text(self):
        """
        Gets the text of a document
        :returns: str -- the text of a document
        """
        return self.text


    def get_tokens(self):
        """
        Gets the tokens of a document
        :returns: str -- the tokens of a document
        """
        return self.tokens


    def get_title(self):
        """
        Gets the title of a document
        :returns: str -- the title of a document
        """
        return self.title


    def get_vector(self):
        """
        Gets the vector representing a document
        :returns: str -- the vector representing a document
        """
        return self.vector




    #--------------------------------------------------------------------------------------------------------
    # SETTERS

    def set_tokens(self,tokens):
        """
        Sets the tokens of the document
        Sets the text of the document too by joining the tokens
        :param list of str tokens: an input list of tokens
        :sets: self.text
        :sets: self.tokens
        """
        self.tokens = tokens
        self.text = " ".join(self.tokens)


    def set_text(self,text):
        """
        Sets the text of the document
        Sets the token of the document too by tokenizing the text
        :param str text: an input text
        :sets: self.text
        :sets: self.tokens
        """
        self.text = text
        self.tokens = nlp_utils.tokenize(self.text)


    def set_vector(self,vector):
        """
        Sets the vector representing a document
        :param vector: an input vector
        :sets: self.vector
        """
        self.vector = vector


    def set_sentiment(self,polarity):
        """
        Sets the sentiment and polarity of a document at the same time
        From the polarity calculate the sentiment class (positive, neutral, negative)
        :param float polarity: the sentiment polarity between -1 and 1
        :sets: float -- self.polarity
        :sets: str -- self.sentiment
        """
        self.polarity = polarity
        self.sentiment = self.convert_polarity_to_sentiment(polarity)




    def set_language(self,language = "english",detect = False):
        """
        Sets the language of a document
        """
        if detect or language is None:
            self.language = self.detect_language()
        else:
            self.language = language




    #--------------------------------------------------------------------------------------------------------
    # FILTERING


    def filter(self,stop_words = False,punctuation = False,number = False,
               stop_vocab = None,strict_number = True,
               min_length = 2,max_length = None,
               language = None,**kwargs):

        # Find the method to filter
        method = "spacy" if self.with_spacy else "nltk"

        # Filter the tokens
        tokens = [token for token in self.tokens if not nlp_utils.filter_token(
                    token = token,method = method,
                    stop_words = stop_words,punctuation = punctuation,number = number,
                    stop_vocab = stop_vocab,strict_number = strict_number,
                    min_length = min_length,max_length = max_length,
                    language = language,**kwargs)]

        # Sets the tokens
        self.set_tokens(tokens)





    #--------------------------------------------------------------------------------------------------------
    # PROCESSING

    def lower(self):
        """
        Sets the text of the document to lowercase 
        """
        self.set_tokens([token.lower() for token in self.tokens])


    def upper(self):
        """
        Sets the text of the document to uppercase 
        """
        self.set_tokens([token.upper() for token in self.tokens])



    def lemmatize(self):
        """
        Lemmatize the tokens of the document
        """
        self.set_tokens([nlp_utils.lemmatize(token) for token in self.tokens])


    def translate(self,to = "english",inplace = True,with_deepl = True):
        """
        Translate the text of the document
        """

        if with_deepl:
            nltk_lang = to
            deepl_lang = LANGUAGES_NLTK_BLOBDEEPL[nltk_lang].upper()
            translation = pydeepl.translate(self.text, deepl_lang)

        else:
            nltk_lang = to
            blob_lang = LANGUAGES_NLTK_BLOBDEEPL[nltk_lang]
            translation = TextBlob(self.text).translate(to = blob_lang).raw

        if inplace:
            self.set_text(translation)
        else:
            return translation



    def correct_mispelling(self,inplace = True):
        """
        Correct the mispelling errors in the text of the document 
        """
        correction = TextBlob(self.text).correct().raw
        if inplace:
            self.set_text(correction)
        else:
            return correction





    #--------------------------------------------------------------------------------------------------------
    # ANALYSIS


    def count_words(self):
        """
        Count the unique tokens in the document
        """
        return nlp_utils.count_words(self.tokens)



    def detect_language(self):
        """
        Detect automatically the language of the document
        Could be https://pypi.python.org/pypi/cld2-cffi
        """
        blob_lang = TextBlob(self.text).detect_language()

        if blob_lang in LANGUAGES_BLOBDEEPL_NLTK:
            nltk_lang = LANGUAGES_BLOBDEEPL_NLTK[blob_lang]
            return nltk_lang
        else:
            return blob_lang



    #--------------------------------------------------------------------------------------------------------
    # SENTIMENT ANALYSIS



    def detect_sentiment_textblob(self):
        """
        Detects the sentiment of the document using the TextBlob library
        The sentiment polarity is between -1 and 1 with -1 (negative) 0 (neutral) and 1 (positive)

        :returns: float -- the sentiment polarity between -1 and 1
        """
        lang = self.detect_language()
        text = self.text if lang == "english" else self.translate(to = "english",inplace = False)
        sentiment = TextBlob(text).correct().sentiment.polarity
        self.sentiment = sentiment
        return sentiment




    def detect_sentiment_vader(self,vader = None):
        """
        Detects the sentiment of the document using the NLTK library and the VADER algorithm
        The sentiment polarity is between -1 and 1, with -1,-0.5 (negative) -0.5,0.5 (neutral) and 0.5,1 (positive)

        Source : 
        - https://github.com/cjhutto/vaderSentiment
        - http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf

        :returns: float -- the sentiment polarity between -1 and 1
        """
        if vader is None:
            vader = SentimentIntensityAnalyzer()

        return vader.polarity_scores(self.text)["compound"]






    def detect_sentiment(self,method = "vader",inplace = True,**kwargs):
        """
        Detects the sentiment of the document
        The sentiment polarity is between -1 and 1

        :param str method: the algorithm used to detect the sentiment, either "vader" or "textblob"
        :param bool inplace: if ``False`` returns the polarity and class, in any case save them as attributes of the document
        :param kwargs: extra arguments can be passed to the vader detector such as the sentiment engine 
        :returns: tuple (float,str) -- of the sentiment polarity and class
        """


        # Detect the polarity
        if method == "vader":
            polarity = self.detect_sentiment_vader(**kwargs)
        elif method == "textblob":
            polarity = self.detect_sentiment_textblob()
        else:
            raise ValueError("method must be 'vader' or 'textblob'")


        # Save the sentiments as attributes
        self.set_sentiment(polarity)

        # Return or save the result
        if not inplace:
            return polarity




    def convert_polarity_to_sentiment(self,polarity):
        """
        Converts a polarity (float between -1 and 1) to a sentiment class (positive, neutral or negative)
        :param float polarity: the sentiment input between -1 and 1
        :returns: str -- the sentiment class ("positive","neutral", or "negative")
        """
        if polarity >= 0.5:
            return "positive"
        elif polarity > -0.5:
            return "neutral"
        else:
            return "negative"

            
    #--------------------------------------------------------------------------------------------------------
    # KEYWORD EXTRACTION (HANDMADE RAKE PROCESS WITH GREAT PERSPECTIVES OF AMELIORATION)

    def _generate_candidate_keywords(self, sentences):
        phrase_list = []
        words = []
        for sentence in sentences:
            words.append(list(map(lambda x: "|" if x in self.stopwords else x,
                        nltk.word_tokenize(sentence.lower()))))
        phrase = []
        for i in range(len(words)):
            for word in words[i]:
                    if word == "|" or isPunct(word):
                        if len(phrase) > 0:
                            phrase_list.append(phrase)
                            phrase = []
                    else:
                        phrase.append(word)

        return phrase_list



    def _calculate_word_scores(self, phrase_list):
        word_freq = nltk.FreqDist()
        word_degree = nltk.FreqDist()
        for phrase in phrase_list:
            degree = len(list(filter(lambda x: not isNumeric(x), phrase))) - 1
            for word in phrase:
                word_freq[word] += 1
                word_degree[word] += degree # other words
        for word in word_freq.keys():
            word_degree[word] = word_degree[word] + word_freq[word] 
        word_scores = {}
        for word in word_freq.keys():
            word_scores[word] = word_degree[word] / word_freq[word]
        return word_scores



    def _calculate_phrase_scores(self, phrase_list, word_scores):
        phrase_scores = {}
        for phrase in phrase_list:
            phrase_score = 0
            for word in phrase:
                    phrase_score += word_scores[word]
                    phrase_scores[" ".join(phrase)] = phrase_score

        return phrase_scores



    def extract_keyword(self, number = 5, keep_scores = False):
        self.get_text()
        self.stopwords = set(nltk.corpus.stopwords.words())
        sentences = nltk.sent_tokenize(self.text)
        phrase_list = self._generate_candidate_keywords(sentences)
        word_scores = self._calculate_word_scores(phrase_list)
        sorted_word_scores = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)

        if keep_scores:
            return list(sorted_word_scores[0:number])
        else:
            return list(map(lambda x: x[0], sorted_word_scores[0:number]))



    def extract_keysentence(self, number = 5, keep_scores = False): 
        self.get_text()
        self.stopwords = set(nltk.corpus.stopwords.words())
        sentences = nltk.sent_tokenize(self.text)
        phrase_list = self._generate_candidate_keywords(sentences)
        word_scores = self._calculate_word_scores(phrase_list)
        sorted_word_scores = sorted(word_scores.items(), key=operator.itemgetter(1), reverse=True)
        phrase_scores = self._calculate_phrase_scores(phrase_list, word_scores)
        sorted_phrase_scores = sorted(phrase_scores.items(), key=operator.itemgetter(1), reverse=True)
        if keep_scores:
            return list(sorted_phrase_scores[0:number])
        else:
            return list(map(lambda x: x[0], sorted_phrase_scores[0:number]))























#=============================================================================================================================
# CORPUS CLASS
#=============================================================================================================================





class Corpus(object):
    def __init__(self,texts = None,titles = None,tokens = None,documents = None,
                    default_language = None,max_documents = None,with_spacy = False,verbose = 1):


        # DEFAULT PARAMETERS
        self.is_title_id = titles is None
        self.verbose = verbose
        self.with_spacy = with_spacy
        self.sentiments_detected = False


        # INITIALIZATION BLOCK
        # Creating the documents
        if documents is None:

            # Existing data
            if texts is not None or tokens is not None:

                # Selecting a subset of documents
                if max_documents is not None:
                    corpus_length = min(len(texts) if texts is not None else len(tokens),max_documents)
                else:
                    corpus_length = len(texts)
                    
                self.documents = []

                # Iterate over each document
                for i in range(corpus_length):
                    title = str(i) if self.is_title_id else titles[i]
                    if texts is None:
                        document = Document(title = title,tokens = tokens[i],default_language = default_language)
                    elif tokens is None:
                        document = Document(title = title,text = texts[i],default_language = default_language)
                    else:
                        raise ValueError("You have to provide a text dataset")

                    self.documents.append(document)
                    self.print("\r[{}/{}] Document loaded in the corpus model".format(i+1,corpus_length),end = "")


            # Empty data
            else:
                self.print(">> Initialized an empty corpus model")
                

        # Reloading the documents    
        else:
            self.documents = documents






    def print(self,message,**kwargs):
        if self.verbose: print(message,**kwargs)



    def __repr__(self):
        verbose = """
        The corpus contains : 
        - {} documents""".format(len(self.documents))

        try:
            verbose += """
            - {} words
            - {} unique word occurences
            - Each document contains on average {} words among which {} are unique
            """.format(len(self.get_all_tokens()),len(self.get_unique_tokens()),
                int(np.mean([len(d.tokens) for d in self.documents])),
                int(np.mean([len(d.get_unique_tokens()) for d in self.documents])))
        except Exception as e:
            print(e)

        return verbose

    def __str__(self):
        return self.__repr__()


    def __iter__(self):
        return iter(self.documents)


    def __getitem__(self,key):
        return self.documents[key]


    def __len__(self):
        return len(self.documents)



    def use_spacy(self):
        pass





    #--------------------------------------------------------------------------------------------------------
    # IO FUNCTIONS - LOADING DATA


    def load_dictionary_of_list_of_tokens(self,dictionary):
        titles,tokens = zip(*dictionary.items())
        self.__init__(titles = titles,tokens = tokens)

    def load_dictionary_of_list_of_texts(self,dictionary,process = True):
        titles,texts = zip(*dictionary.items())
        self.__init__(titles = titles,texts = texts,process = process)

    def load_json_of_tokens(self,json_path):
        dictionary = json.loads(open(json_path,"r").read())
        self.load_dictionary_of_list_of_tokens(dictionary)

    def load_json_of_texts(self,json_path,process = True):
        dictionary = json.loads(open(json_path,"r").read())
        self.load_dictionary_of_list_of_texts(dictionary,process = process)



    def load_from_mongodb(self,database,collection,host,text_field,title_field = None,port = 27017,max_documents = None):
        """
        Load a corpus from an existing MongoDB Database

        :param str database: the name of the database
        :param str collection: the name of the collection
        :param str host: the mongoDB host
        :param str text_field: the key containing the text for a document
        :param str title_field: the key containting the title for a document
        :param str port: the mongoDB port
        :param int max_documents: to retrieve a given number of documents, all documents if None

        """
        
        # Connection to the MongoDB client and collection
        collection = MongoClient(host = host,port = port)[database][collection]

        # Initialize the containers
        texts = []
        titles = [] if title_field is not None else None

        # Yield the documents from the database
        max_documents_args = {"limit":max_documents} if max_documents is not None else {}

        for document in tqdm(collection.find(**max_documents_args)):
            texts.append(document[text_field])
            if title_field is not None:
                titles.append(document[title_field])

        
        # Initialize the corpus with the texts and titles
        self.__init__(texts = texts,titles = titles)








    #--------------------------------------------------------------------------------------------------------
    # IO FUNCTIONS - SAVING DATA

    def save_as_json_of_tokens(self,json_path):
        dictionary = {document.title:document.tokens for document in self.documents}
        with open(json_path,'w') as file:
            json.dump(dictionary,file)

    def save_as_json_of_texts(self,json_path):
        pass

    def save_in_mongodb(self,mongo_client):
        pass

        

    def to_DataFrame(self,with_tokens = True,with_clusters = False):
        data = self.get_dictionary_for_dataframe(with_tokens,with_clusters)
        columns = ["title","text"]
        if with_tokens: columns.append("tokens")
        if with_clusters: columns.append("cluster")

        data = pd.DataFrame(data,columns = columns)

        if self.is_title_id:
            data.drop("title",axis = 1,inplace = True)

        return data

        





    #--------------------------------------------------------------------------------------------------------
    # GETTERS


    def get_unique_tokens(self):
        return list(set([token for document in self.documents for token in document.get_unique_tokens()]))


    def get_tokens(self):
        return [document.get_tokens() for document in self.documents]


    def get_titles(self):
        return [document.get_title() for document in self.documents]




    def get_all_tokens(self,cluster = None):
        if cluster is None:
            return [token for document in self.documents for token in document.tokens]
        else:
            return [token for document in self.documents for token in document.tokens if document.cluster == cluster]


    def get_clusters(self):
        return [document.cluster for document in self.documents]

    def get_text(self,flatten = False):
        if flatten:
            return " ".join(self.get_text(flatten = False))
        else:
            return [document.get_text() for document in self.documents]

    def get_dictionary_of_texts(self):
        return {document.title:document.text for document in self.documents}


    def get_dictionary_for_dataframe(self,with_tokens = True,with_clusters = False):

        def create_dictionary(document,with_tokens = True,with_clusters = False):
            dictionary = {"title":document.title,"text":document.text}
            if with_tokens:
                dictionary["tokens"] = document.tokens
            if with_clusters:
                dictionary["cluster"] = document.cluster
            return dictionary

        return [create_dictionary(document,with_tokens,with_clusters) for document in self.documents]


    def get_data(self,top_n_words = None):
        data = []
        for document in self.documents:
            document_data = {}
            document_data["title"] = document.title
            document_data["cluster"] = document.cluster

            if top_n_words is not None:
                document_data["top {} words".format(top_n_words)] = ", ".join(document.count_words().index[:top_n_words])

            data.append(document_data)

        return pd.DataFrame(data).set_index("title")



    def get_sentiments(self):
        return [document.sentiment for document in self.documents]



    def get_polarities(self):
        return [document.polarity for document in self.documents]




    #--------------------------------------------------------------------------------------------------------
    # FILTERING

    def filter(self,stop_words = False,punctuation = False,number = False,
               stop_vocab = None,strict_number = True,
               min_length = 2,max_length = None,
               language = None,**kwargs):
        """
        Filtering function which removes token from the texts in the corpus
        Applies iteratively to all documents in the corpus
        Arguments : 
            - language (string or list of strings), languages supported by nltk, needed to delete the stopwords
            - vocabulary (a list of strings), a given list of words to look up and remove
            - numeric (boolean), to remove tokens containing numbers
            - punctuation (boolean), to remove tokens containing punctuation
            - min_length (int), to remove tokens shorter than min_length
            - max_length (int), to remove tokens longer than max_length
        Returns : None
        """

        # ITERATING OVER ALL DOCUMENTS
        for i,document in enumerate(self.documents):
            self.print('\r[{}/{}] Filtering documents '.format(i+1,len(self.documents)),end = "")
            document.filter(stop_words = stop_words,
                punctuation = punctuation,
                number = number,
                stop_vocab = stop_vocab,
                strict_number = strict_number,
                min_length = min_length,max_length= max_length,
                language = language,
                **kwargs)


        self.print("")




    def clean(self,language = "english"):
        print(">> Cleaning the corpus")
        self.lower()
        self.filter(punctuation = True,numeric = True)
        self.bigrams = self.token_collocation()
        self.trigrams = self.token_collocation()
        self.lemmatize()
        self.filter(language = language,max_length = 30)






    def filter_on_tfidf_score(self,threshold = 0.05,n_total_words = 1000,n_top_words = None):

        data = []

        X = self.X_tfidf.drop("cluster",axis = 1) if "cluster" in self.X_tfidf.columns else self.X_tfidf

        total_words = list(X.sum(axis = 0).sort_values(ascending = False).iloc[:n_total_words].index)

        
        X = self.X_tfidf.loc[:,total_words]


        for i,document in enumerate(tqdm(self.documents,desc = "Filtering on tfidf score")):
            top_words = X.iloc[i].sort_values(ascending = False)

            if n_top_words is not None:
                top_words = list(top_words.iloc[:n_top_words].index)
            else:
                top_words = list(top_words.loc[top_words > threshold].index)

            data.append(top_words)

        df = pd.DataFrame(columns = ["Top Words"])
        df["Top Words"] = data

        return df




    def filter_on_language(self,language):
        documents = []
        for document in tqdm(self):
            if document.language == language:
                documents.append(document)
        self.documents = documents








    #--------------------------------------------------------------------------------------------------------
    # PROCESSING




    def lower(self):
        for i in range(len(self.documents)):
            document = self.documents[i]
            self.print("\r[{}/{}] Lowering document {}".format(i+1,len(self.documents),document.title),end = "")
            document.lower()

        self.print("")


    def lemmatize(self):
        for i in range(len(self.documents)):
            document = self.documents[i]
            self.print("\r[{}/{}] Lemmatizing document {}".format(i+1,len(self.documents),document.title),end = "")
            document.lemmatize()


        self.print("")




    def apply_token_collocation(self):
        sentences = self.get_tokens()
        model = nlp_utils.token_collocation_model(sentences)
        count = nlp_utils.count_token_collocation_model(model)
        sentences = Phraser(model)[sentences]
        for i,tokens in enumerate(sentences):
            self.print('\r[{}/{}] Applying token collocation model on documents '.format(i+1,len(self.documents)),end = "")
            self.documents[i].set_tokens(tokens)
        return count
        


    def translate(self,to = "english",with_deepl = True):
        for i in range(len(self.documents)):
            document = self.documents[i]
            language = document.language
            self.print("\r[{}/{}] Translating document with {}".format(i+1,len(self.documents),"DeepL" if with_deepl else "TextBlob"),end = "")

            if language != to:
                document.translate(to = to,with_deepl = with_deepl)

        self.print("")






    #--------------------------------------------------------------------------------------------------------
    # SEMANTIC REPRESENTATION MODELS


    def build_tfidf_model(self,inplace = False,as_matrix = False):
        """
        Creates a vectorized representation of the corpus using a TF-IDF model
        Will also save the vectorizer as attribute in .vectorizer
        Arguments : 
            - inplace (boolean) :  to either returns the matrix or save it as argument
            - as_matrix (boolean) :  to return either a Pandas DataFrame or a Numpy matrix
        Returns : 
            X_tfidf the matrix tfidf representation of the corpus

        """
        self.print(">> Building TF-IDF model ... ",end = "")
        self.vectorizer = TfidfVectorizer()
        X_tfidf = self.vectorizer.fit_transform(self.get_text()).todense()
        if not as_matrix:
            X_tfidf = pd.DataFrame(X_tfidf,columns = [x[0] for x in sorted(self.vectorizer.vocabulary_.items(),key = lambda x : x[1])])
        self.print("ok")

        if inplace:
            self.X_tfidf = X_tfidf
        else:
            return X_tfidf





    def build_word2vec_model(self,size = 200,min_count = 5,iter = 5):
        """
        Creates a Word2Vec representation of the words in the corpus
        Arguments : 
            - size (int) : the number of components in the vector
            - min_count (int) : the minimum number of occurences to consider a token
            - iter (int) : the number of epochs to train the word2vec model
        Creates: the attribute w2v (gensim Word2Vec object)
        Returns: None
        """

        self.print(">> Building Word2Vec model ... ",end = "")

        # Initialize the model
        self.w2v = Word2Vec(size=size, min_count=min_count,iter = iter)

        # Build the vocab
        self.w2v.build_vocab(self.get_tokens())

        # Train the model for n epochs
        self.w2v.train(self.get_tokens(),total_examples=self.w2v.corpus_count, epochs=self.w2v.iter)
        self.print("ok")





    def build_doc2vec_model(self,inplace = True):
        # TFIDF MODEL
        if hasattr(self,"X_tfidf") and isinstance(self.X_tfidf,pd.DataFrame):
            X_tfidf = self.X_tfidf
        else:
            X_tfidf = self.build_tfidf_model(inplace = False,as_matrix = False)

        # WORD2VEC MODEL
        if not hasattr(self,"w2v"):
            self.build_word2vec_model(iter = iter)


        # MERGE THE MODELS TO CREATE A DOC2VEC MODEL
        self.print(">> Building Doc2Vec model ... ",end = "")
        filtered_vocab = [x for x in X_tfidf.columns if x in self.w2v]
        X_tfidf = X_tfidf[filtered_vocab].as_matrix()
        P_w2v = self.get_word2vec_matrix(filtered_vocab,as_matrix = True)
        X_d2v = np.dot(X_tfidf,P_w2v)
        self.print("ok")


        # RETURN THE MODEL
        if inplace:
            self.X_d2v = X_d2v
        else:
            return X_d2v





    def get_word2vec_matrix(self,vocab,as_matrix = False):
        # PREREQUISITES
        assert hasattr(self,"w2v")

        # BUILD MATRIX
        filtered_vocab = [x for x in vocab if x in self.w2v]
        matrix = np.array([self.w2v[x] for x in filtered_vocab])

        if as_matrix:
            return matrix
        else:
            return pd.DataFrame(matrix,index = filtered_vocab)







    def get_model(self,model = "doc2vec"):
        # ASSERTIONS
        assert model in ["doc2vec","tfidf"]

        if model == "doc2vec":
            if hasattr(self,"X_d2v"):
                return self.X_d2v
            else:
                return self.build_doc2vec_model(inplace = False)

        elif model == "tfidf":
            if hasattr(self,"X_tfidf"):
                return self.X_tfidf
            else:
                return self.build_tfidf_model(inplace = False,as_matrix = True)
        else:
            pass





    #--------------------------------------------------------------------------------------------------------
    # PLOTTING

    def plot_number_of_words_distributions(self,unique = False):
        plt.figure(figsize = (8,4))
        if not unique:
            plot_data = [len(document.tokens) for document in self.documents]
        else:
            plot_data = [len(document.get_unique_tokens()) for document in self.documents]
        plt.hist(plot_data,bins = 20)
        plt.show()





    def plot_TSNE(self,model = "doc2vec",colors = None,n_components = 2,output = False,metric = "cosine",with_plotly = True,html = None,**kwargs):
        self.print(">> Plotting t-SNE representation")

        # GET THE MODEL
        X = self.get_model(model)

        # BUILD THE TSNE MODEL
        distance_matrix = pairwise_distances(X,metric = metric)
        model = TSNE(n_components = n_components,metric = "precomputed",random_state = 1,**kwargs)
        tsne_points = model.fit_transform(distance_matrix)
        x_tsne,y_tsne = tsne_points[:,0],tsne_points[:,1]

        # COLORS
        if colors is not None:
            extract_color = lambda x : "rgba({},{},{},1)".format(*list(map(lambda x : int(x*255),x))) 
            unique_values = list(set(colors))
            palette = [extract_color(c) for c in sns.color_palette("Paired",n_colors= len(unique_values))]
            colors_mapping = {unique_values[i]:palette[i] for i in range(len(unique_values))}
            colors = [colors_mapping[m] for m in colors]

        else:
            colors = "rgba(166,206,227,1)"


        # WITH PLOTLY
        if with_plotly:
            # titles = [x+"\n"+y[:200] for x,y in zip(self.get_titles(),self.get_text())]
            titles = self.get_titles()
            trace = go.Scatter(
                x = x_tsne,
                y = y_tsne,
                mode = 'markers',
                marker = dict(color = colors,line = dict(width = 1),size = 9),
                text = titles
            )
            layout = dict(title = 't-SNE representation',
                    width = 1200,
                    height = 700,
                    yaxis = dict(zeroline = False,showgrid = False,showticklabels = False),
                    xaxis = dict(zeroline = False,showgrid = False,showticklabels = False)
                    )
            data = [trace]

            if html is None:
                iplot({"data":data,"layout":layout})
            else:
                plot({"data":data,"layout":layout},filename = html)

        # WITH MATPLOTLIB
        else:
            plt.figure(figsize = (8,8))
            plt.scatter(x_tsne,y_tsne,c = colors,s=25)
            plt.show()

        # RETURN VALUES
        if output:
            return x_tsne,y_tsne



            


    def plot_word_cloud(self,width = 600,height = 300,max_words = 50,figsize = (18,10),**kwargs):
        fig = nlp_utils.plot_word_cloud(self.get_text(flatten = True),
                        width = width,height = height,
                        max_words = max_words,figsize = figsize,**kwargs)
        return fig







    #--------------------------------------------------------------------------------------------------------
    # ANALYSIS



    def count_words(self):
        return nlp_utils.count_words(self.get_all_tokens())



    def show_language_distribution(self,display = True,with_plotly = False):
        langs = [document.language for document in self.documents]
        df = pd.DataFrame(langs,columns = ["language"])
        df["count"] = 1
        df = df.groupby("language").sum()

        if display:
            if with_plotly:
                pass
            else:
                df.plot(kind = "bar",figsize = (12,4))
                plt.show()
        else:
            return df



    def compute_word_importance(self,sort = "average tfidf"):
        df = self.count_words()
        average_tfidf = pd.DataFrame(self.X_tfidf.transpose().mean(axis = 1),columns = ["average tfidf"])
        df = df.join(average_tfidf)
        df = df.sort_values(sort,ascending = False)
        return df


        



    #--------------------------------------------------------------------------------------------------------
    # SENTIMENT ANALYSIS



    def detect_sentiment(self,method = "vader"):
        """
        Detects sentiment for all documents in a corpus
        :param str method: the algorithm used to detect the sentiment, either "vader" or "textblob"
        :sets: self.sentiments_detected to True to avoid recomputing the detection
        :sets: the sentiment and polarity for every document in the corpus
        """

        # Prepare the extra args
        # For Vader the computation is optimized if we keep the sentiment engine from reloading
        if method == "vader": 
            vader = SentimentIntensityAnalyzer()
            kwargs = {"vader":vader}
        else:
            kwargs = {}

        # Iterate over every documents
        for document in tqdm(self.documents):
            document.detect_sentiment(method = method,inplace = True,**kwargs)

        # Set an argument for future operations
        self.sentiments_detected = True








    def show_sentiment_distribution(self,with_plotly = False,kind = "pie",on_notebook = True,method = "vader"):
        assert kind in ["pie","bar","hist"]

        # If sentiments never detected, detect them
        if not self.sentiments_detected:
            self.detect_sentiment(method = method)


        # Find the sentiments for all documents
        sentiments = self.get_polarities() if kind == "hist" else self.get_sentiments()
        sentiments = pd.DataFrame(sentiments,columns = ["sentiment"])

        # Discretize the sentiment for pie and bar representations and aggregate
        if kind != "hist":
            sentiments["count"] = 1
            sentiments = sentiments.groupby("sentiment", as_index=False).sum()


        if with_plotly:
            if kind == 'pie':
                fig = { 

                "data": [{
                            'values': [value for value in sentiments["count"]],
                            'labels': [label for label in sentiments["sentiment"]],
                            'type': kind,
                            'hole': 0.3,
                            'name': "Sentiment Distribution",
                            "hoverinfo":"label+percent+name",
                            }] , 
                "layout": {
                'title': "Sentiment Distribution",
                }
                }
            elif kind == "bar":
                fig = { 

                "data": [{
                            'y': [value for value in sentiments["count"]],
                            'x': [label for label in sentiments["sentiment"]],
                            'type': kind,
                            'name': "Sentiment Distribution",
                            "hoverinfo":"label+percent+name",
                            }] , 
                "layout": {
                'title': "Sentiment Distribution",
                }
                }

            else:
                data = go.Histogram( x = sentiments["sentiment"],xbins=dict(size=0.1), name = 'Sentiment Distribution', hoverinfo = "name")
                layout = go.Layout(title ="Sentiment Distribution")
                fig = go.Figure(data=[data], layout=layout)

            if on_notebook:
                iplot(fig)
            else:
                return fig
        else:
            if kind == "pie": 
                args = {"subplots":True,"colors":["red","grey","green"],"figsize":(8,8)}
            elif kind == "bar":
                args = {"colors":[["red","grey","green"]],"figsize":(12,4)}
            else:
                args = {"figsize":(12,4)}

            sentiments.plot(kind = kind,**args)
            plt.show()



    def show_sentiment_distribution_by_document(self):
        # Find the sentiments for all documents
        polarities = self.get_polarities()
        polarities = pd.DataFrame(polarities,columns = ["polarity"])
        polarities["text extract"] = [document.get_text()[:150] for document in self.documents]
        return polarities[["text extract","polarity"]].sort_values("polarity",ascending = False)







    #--------------------------------------------------------------------------------------------------------
    # NETWORK REPRESENTATION




    def build_word_network(self,similarity = "cooccurrences",threshold = None,top_n_words = 200,inplace = True):
        """
        Build a graph network representation of the words in the corpus
        Nodes are the words in the corpus,
        The nature of the links between the nodes depends on the similarity measured between words

        Similarity : 
            - "cooccurrences" will look for words that appear together in a same document in the corpus, 
              the weight of a link will be the number of documents in which two tokens appear together
            - "cosine" will calculate the cosine similarity between the Word2Vec vector representation of the two words

        The construction of the network uses the network part of the library


        Arguments:
            - similarity (string) : either "cooccurrences" or "cosine", raises an error otherwise
            - threshold (float) : the min threshold to consider a link with the Word2Vec cosine similarity
            - top_n_words (int) : to decrease the total number of words considered, the filter is on the top tfidf scores
            - inplace (bool) : to either save as an attribute of the corpus (True) or return the graph (False)

        Returns or creates: a networkx Graph() object
        """


        # ASSERTIONS
        assert similarity in ["cooccurrences","cosine"]


        # CO OCCURENCES SIMILARITY
        if similarity == "cooccurrences":

            if top_n_words is None:      
                tokens = self.get_tokens()
            else:
                tokens = self.filter_on_tfidf_score(n_total_words = top_n_words)
                tokens = [X.iloc[i]["Top Words"] for i in range(len(X))]
                tokens = [t for t in tokens if len(t)>=2]

            graph = network_utils.build_network_from_list_of_occurences(tokens,min_count = 2,min_degree = 1)



        # COSINE SIMILARITY BETWEEN WORD2VEC VECTORS 
        elif similarity == "cosine":

            # Creating the vocab and limiting it to the top_n_words by tfidf score
            # vocab = self.compute_word_importance(sort = "average tfidf")
            df = self.count_words()
            average_tfidf = pd.DataFrame(self.X_tfidf.transpose().mean(axis = 1),columns = ["average tfidf"])
            df = df.join(average_tfidf)
            vocab = df.sort_values("average tfidf",ascending = False)


            if top_n_words is not None: vocab = vocab.iloc[:top_n_words]
            vocab = list(vocab.index)

            # Safety check so that all the vocab has a vector              
            vocab = [x for x in vocab if x in self.w2v]

            # Creating the node data
            nodes_data = [{"name":x,"data":self.w2v[x]} for x in vocab]

            # Defining the cosine similarity function
            similarity_function = lambda X,Y:network_utils.cosine_similarity(X,Y)

            # Create the graph through the networks utils
            graph = network_utils.build_network_on_similarity(nodes_data,similarity_function,threshold = threshold)




        # RETURNING THE GRAPH
        if inplace:
            self.word_network = graph
        else:
            return graph






        


    def build_document_network(self,similarity = "cooccurrences_norm",extract_function = None,threshold = None,inplace = True):
        """
        Arguments: the similarity function (string or function) used to compute the similarity between documents
            - "cooccurences_norm" will calculate the number of tokens shared by 2 documents (normalized)        
            - "cooccurences" will calculate the number of tokens shared by 2 documents (not normalized)
            - "cosine" will calculate the cosine similarity between 2 documents vectors represented with Doc2Vec

        """

        # Define a function that extracts the correct data from the document to compute the similarity
        def extract_data(document,similarity):
            if isinstance(similarity,str):
                if similarity in ["cooccurrences","cooccurrences_norm"]:
                    return document.get_tokens()
                elif similarity in ["cosine"]:
                    return document.get_vector()
                else:
                    pass
            else:
                return extract_function(document)



        # Define the similarity function
        def set_function(similarity):
            if isinstance(similarity,str):
                if similarity == "cooccurrences":
                    return lambda X,Y : network_utils.cooccurrences_similarity(X,Y,norm = False)
                elif similarity == "cooccurrences_norm":
                    return lambda X,Y : network_utils.cooccurrences_similarity(X,Y,norm = True)
                elif similarity == "cosine":
                    return lambda X,Y : network_utils.cosine_similarity(X,Y)
            else:
                return similarity

        similarity_function = set_function(similarity)


        # Create the data in a suitable format to be passed in the networks utils
        nodes_data = [{"name":document.get_title(),"data":extract_data(document,similarity)} for document in self.documents]

        # Create the network graph
        graph = network_utils.build_network_on_similarity(nodes_data,similarity_function,threshold = threshold)
        


        if inplace:
            self.document_network = graph
        else:
            return graph








    def export_network(self,file_path,level = "document",graph = None):
        if graph is None:
            assert level in ["document","word"]
            graph = self.document_network if level == "document" else self.word_network

        network_utils.export_graph_to_gml(graph,file_path)











    def plot_network(self,level = "document",graph = None,**kwargs):
        if graph is None:
            assert level in ["document","word"]
            graph = self.document_network if level == "document" else self.word_network

        network_utils.plot_graph(graph,**kwargs)














    #--------------------------------------------------------------------------------------------------------
    # CLUSTERING


    def clustering(self,model = None, n_clusters = 10):
        """
        The clustering function takes all the documents in the corpus and clusterizes them. 

        Takes as input : 
            - a raw sklearn clustering model
            - By default the model is KMeans, in that case you can choose the number of clusters you want

        Creates: self.X a dataframe with the tfidf vector and the calculated cluster
        Updates: the cluster attribute for each document in the corpus
        Returns: None
        """

        # LAUNCH THE TIMER
        timer = Timer()
        self.print(">> Clustering on the corpus ...")

        # CREATE THE CLUSTERING MODEL
        if model is None:
            model = KMeans(n_clusters = n_clusters)

        # CREATE A TFIDF DATASET FROM THE CONTENT OF THE DOCUMENTS
        self.X = self.build_tfidf_model()

        # PREDICT THE CLUSTERS WITH THE CLUSTERING MODEL
        try:
            clusters = model.fit_predict(self.X)
        except TypeError:
            clusters = model.fit_predict(self.X.toarray())


        # SET THE CLUSTER NUMBER IN THIS MATRIX
        self.X["cluster"] = clusters

        # SET THE CLUSTER NUMBER FOR EACH DOCUMENT AS A DOCUMENT ATTRIBUTE
        for i,document in enumerate(self.documents):
            setattr(document,"cluster",clusters[i])

        # END THE TIMER FOR THE CLUSTERING PROCESS
        timer.end("clustering")











    def describe_clusters(self,top_n_words = 5,tfidf = True):
        """
        The describe clusters function will be used to show the top words in each previously computed cluster
        Takes as input: 
            - The number of words to show
            - The boolean tfidf = True to show top words ordered by tfidf score or by count

        Prints: the clusters and their top words
        Returns: a Pandas DataFrame with the cluster description
        """

        # GET THE DATA FROM THE CORPUS
        data = self.get_data()

        # GET THE CLUSTERS
        clusters = sorted(data["cluster"].unique())

        # INITIALIZE A DICTIONARY THAT WILL BE CONVERTED TO DATAFRAME LATER
        description = {'cluster': [],
                        'top_{0}_words'.format(top_n_words): []}

        # AGGREGATE THE AVERAGE TFIDF SCORE FOR EACH CLUSTER
        clusters_tfidf_scores = self.X.groupby("cluster").mean()

        # ITERATE OVER THE CLUSTERS
        for cluster in clusters:

            # If tfidf display the top tfidf words
            if tfidf:
                tokens = ", ".join(list(clusters_tfidf_scores.loc[cluster].sort_values(ascending = False).index)[:top_n_words])

            # Otherwise display the top counted words
            else:
                tokens = ", ".join(nlp_utils.count_words(self.get_tokens(cluster = cluster,flatten = True)).index[:top_n_words])

            # Updates the dictionary and print
            description['cluster'].append(cluster)
            description['top_{0}_words'.format(top_n_words)].append(tokens)
            self.print(">> Cluster {} - {} documents - top words : {}".format(cluster,len(data.loc[data.cluster == cluster]),tokens))

            

        return pd.DataFrame(description)




    #--------------------------------------------------------------------------------------------------------------
    # SPACY FUNCTIONS TO BE CONVERTED


  




    def LDA(self,n_topics = 20,mm_path = "mm_model.mm",workers = 3,save = None,n_below = 10,n_above = 0.3):
        # Import
        import pyLDAvis
        import pyLDAvis.gensim

        self.start_timer()

        """Compute the Latente Dirichlet Allocation algorithm"""
        print(">> Computing the Latent Dirichlet Allocation algorithm on the corpus")

        # Build a dictionary indexing the tokens in the corpus
        print("... Indexing the tokens in the corpus")
        self.create_dictionary_of_tokens(n_below = n_below,n_above = n_above)

        # Creating a Sparse matrix indexing the corpus
        print("... Creating a Bag of Words representation of the corpus as a sparse matrix")
        MmCorpus.serialize(mm_path,nlp_utils.bow_generator(self.get_tokens(),self.dictionary))
        mm_corpus = MmCorpus(mm_path)

        # Applying LDA algorithm on the corpus
        print("... Applying the Latent Dirichlet Algorithm to detect the topics")
        self.lda = LdaMulticore(mm_corpus,num_topics=n_topics,id2word=self.dictionary,workers=workers)
        lda_prepared = pyLDAvis.gensim.prepare(self.lda, mm_corpus,self.dictionary)

        if save is not None:
            self.lda.save("{}.lda".format(save))
            with open("{}.p".format(save), 'wb') as file:
                pickle.dump(lda_prepared,file)

        self.end_timer()

        return lda_prepared