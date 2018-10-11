#!/usr/bin/env python
# -*- coding: utf-8 -*- 


__author__ = "Theo"


"""--------------------------------------------------------------------
NATURAL LANGUAGE PROCESSING FUZZY MATCHING FUNCTIONS
Grouping various scripts and functions for nlp 

Started on the 02/06/2017


Distance possible: 
Levenshtein distance
Damerau-Levenshtein distance
Needleman–Wunsch algorithm
Spell-checker method
Ratcliff/Obershelp algorithm

http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/


theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy
import time
import re
import string

# NLTK
import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams

# FUZZYWUZZY
from fuzzywuzzy import fuzz,process

# GENSIM
import warnings
import gensim
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models.word2vec import Word2Vec

# SKLEARN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# EKIMETRICS
from ekimetrics.utils.time import sec_to_hms




#=============================================================================================================================
# FAST FUZZY MATCHING
# Optimized version by Theo ALVES DA COSTA
#=============================================================================================================================


def create_ngrams(s,n = 3):
    return ["".join(x) for x in ngrams(s,n)]



def tokenize_ngrams(texts,n = 3,**kwargs):
    return [create_ngrams(text,n = n,**kwargs) for text in texts]




def create_token2vec_model(tokens,size = 100,iter = 500,min_count = 2,**kwargs):
    print(">> Creating Token2Vec model",end = "")
    t2v = Word2Vec(size = size,min_count=min_count,iter = iter,**kwargs)
    t2v.build_vocab(tokens,keep_raw_vocab=True)
    t2v.train(tokens,total_examples=t2v.corpus_count, epochs=t2v.iter)
    print(" - ok")
    return t2v


def create_tfidf_model(texts,tokenizer = lambda x : x.split("|")):
    print(">> Creating TfIdf model",end = "")
    vectorizer = TfidfVectorizer(lowercase=False,tokenizer = tokenizer)
    tfidf = vectorizer.fit_transform(texts).todense()
    tfidf = pd.DataFrame(tfidf,columns = [x[0].upper() for x in sorted(vectorizer.vocabulary_.items(),key = lambda x : x[1])])
    print(" - ok")
    return vectorizer,tfidf





class FastFuzzyMatching(object):
    """
    Fast Fuzzy Matching class
    """

    def __init__(self):
        """
        Initialization
        """
        pass




    #-----------------------------------------------------------------------------------------
    # PREPROCESSING FUNCTIONS


    def _preprocess(self,X,Y = None,auto_detect = False,vocabulary = None,method = "word",max_count = None):

        if vocabulary is None:
            vocabulary = []

        if auto_detect:

            if max_count is None:
                max_count = 20

            frequent_tokens = self.most_frequent_tokens(X,Y,method)
            frequent_tokens = frequent_tokens.loc[frequent_tokens > max_count]
            frequent_tokens = list(frequent_tokens.index)

            vocabulary.extend(frequent_tokens)


        def preprocess(x,vocabulary):
            return re.sub(r'|'.join(map(re.escape, list(string.punctuation) + vocabulary)), '', x).strip().upper()


        X = [preprocess(x,vocabulary) for x in X]

        if Y is not None:
            Y = [preprocess(y,vocabulary) for y in Y]

        return X,Y




    def most_frequent_tokens(self,X,Y = None,method = "trigrams"):
        _,_,tokens = self._tokenize(X,Y,method)
        all_tokens = [token for t in tokens for token in t]
        return pd.Series(all_tokens).value_counts()






    #-----------------------------------------------------------------------------------------
    # HELPER FUNCTIONS

    def _tokenize(self,X,Y = None,method = "trigrams",**kwargs):
        """
        Tokenize the list of strings
        """

        entities = list(X)
        if Y is not None:
            entities.extend(list(Y))

        if type(method) != list:

            if "trigram" in method:
                tokens = tokenize_ngrams(entities,n = 3,**kwargs)
            elif "bigram" in method:
                tokens = tokenize_ngrams(entities,n = 2,**kwargs)
            elif "word" in method:
                tokens = [entity.split() for entity in entities]
            else:
                raise ValueError("Method not understood")

            texts = ["|".join(x) for x in tokens]

        else:

            tokens = [self._tokenize(X,Y,method = m)[2] for m in method]
            tokens = [list(set([token for tokens in method_tokens for token in tokens])) for method_tokens in zip(*tokens)]

            texts = ["|".join(x) for x in tokens]

        return entities,texts,tokens






    def _transform_tfidf(self,texts):
        """
        Transform the dataset to the TfIdf representation of its tokens
        """
        tfidf = self.vectorizer.transform(texts).todense()
        tfidf = pd.DataFrame(tfidf,columns = [x[0].upper() for x in sorted(self.vectorizer.vocabulary_.items(),key = lambda x : x[1])])
        return tfidf




    def _encode_e2v(self,tfidf,verbose = 0):
        """
        Encode the entity to its vectorized representation
        """

        if verbose: print(">> Encoding to Entity2Vec representation",end = "")

        # Select only the common tokens and reorder the matrices in the same token order
        filtered_vocab = [x for x in tfidf.columns if x in self.t2v]
        tfidf = tfidf[filtered_vocab].as_matrix()

        # Create a transfer matrix 
        P_t2v = np.array([self.t2v[x] for x in filtered_vocab])

        # Encode in the entity vectorized space
        e2v = np.dot(tfidf,P_t2v)

        if verbose: print(" - ok")

        return e2v




    def _compute_distances(self,X,Y,metric = "cosine"):
        """
        Compute the distance matrix between encoded representations via "cosine" metric
        """
        return scipy.spatial.distance.cdist(X,Y,metric)






    def _create_nearest_neighbors_model(self,e2v,n = 5):
        """
        Create a Nearest Neighbors model from encoded representation of the entities
        """
        nn = NearestNeighbors(n_neighbors=n, algorithm='auto',metric = "cosine").fit(e2v)
        return nn






    #-----------------------------------------------------------------------------------------
    # FUNCTIONS


    def train(self,X,Y = None,method = "trigrams",size = 100,iter = 500,min_count = 2,tokenizer = lambda x : x.split("|"),**kwargs):
        """
        Training function
        """

        # Start a timer
        s = time.time()

        # Tokenization
        self.entities,texts,tokens = self._tokenize(X,Y,method)

        # Train the token2vec model
        self.t2v = create_token2vec_model(tokens,size = size,iter = iter,min_count = min_count,**kwargs)
        
        # Prepare the TfIdf model
        self.vectorizer,tfidf = create_tfidf_model(texts,tokenizer = tokenizer)

        # Encode the tfidf representation to the lower dimension representation
        self.e2v = self._encode_e2v(tfidf,verbose = 1)

        # Pre train a Nearest neighbors model
        self.nn = self._create_nearest_neighbors_model(self.e2v)

        # End the timer
        e = time.time()
        print("... Training finished in {}".format(sec_to_hms(e-s)))





    def match(self,X,Y = None,method = "trigrams",retrain = False,as_distance_matrix = False,n = 5,threshold = 0.5,**kwargs):
        """
        Fuzzy matching two lists of texts
        """
        _,texts,_ = self._tokenize(X,Y,method)

        # Training phase
        if not hasattr(self,"t2v") or not hasattr(self,"vectorizer") or retrain:
            self.train(X,Y,method,**kwargs)

        # Vectorize
        tfidf = self._transform_tfidf(texts)

        # Encode
        e2v = self._encode_e2v(tfidf)

        # Separate X and Y
        X_e2v = e2v[:len(X)]
        Y_e2v = e2v[len(X):]

        if as_distance_matrix:
            # Compute distances
            distances = self._compute_distances(X_e2v, Y_e2v, 'cosine')
            return distances

        else:
            # Fit a K Nearest Neighbors unsupervised model
            nn = self._create_nearest_neighbors_model(Y_e2v,n = n)

            # Find the best matches with K Nearest Neighbors algorithms
            distances,indices = nn.kneighbors(X_e2v)

            # Format the results
            find_entity = np.vectorize(lambda indice : Y[indice])
            results = np.stack([find_entity(indices),distances],axis = 2)
            columns = [y for b in range(1,n+1) for y in ("match {}".format(b),"score {}".format(b))]
            results = pd.DataFrame(results.reshape((len(results),np.prod(results.shape[1:]))),index = X,columns = columns)
            for col in results.columns:
                if "score" in col:
                    results[col] = results[col].astype(float).round(3)

            return results



            

    def most_similar(self,s,method = "trigrams",as_distance_matrix = False,n = 5,threshold = 0.5):
        """
        Find the most similar dataset in a trained dataset model
        """

        # Create token representation
        _,texts,_ = self._tokenize([s],method = method)

        print(texts,_)
        
        # Vectorize and encode
        tfidf = self._transform_tfidf(texts)
        e2v = self._encode_e2v(tfidf)

        if as_distance_matrix:
            # Compute distances
            distances = self._compute_distances(e2v, self.e2v, 'cosine')
            return distances
        else:

            # Find the best match with pre trained K Nearest Neighbors algorithms
            distances,indices = self.nn.kneighbors(e2v)

            # Format the results
            results = [(self.entities[x[0]],x[1]) for x in zip(indices[0],distances[0]) if x[1] < threshold]
            return results








#=============================================================================================================================
# EXTRACT BEST FUZZY MATCHES
#=============================================================================================================================



def fuzzy_match(X,Y,threshold = 60):
    """
    Fuzzy matching between two elements

    :param list X: the input list or element to be matched
    :param list Y: the target list to be matched on
    :param int or list threshold: the threshold to consider under which the similarity will yield a None
    :returns: pandas.DataFrame -- each element of X matched (or the best match for an single element)
    """

    if type(threshold) != list:

        # Case single element
        if type(X) != list:
            choice = process.extract(X,Y)[0][0]
            match = fuzz.token_sort_ratio(X,choice)
            if match > threshold:
                return choice
            else:
                return None
        else:
            tqdm.pandas(desc="Fuzzy matching at threshold {}%".format(threshold))
            data = pd.DataFrame(X,columns = ["input"])
            data["match_{}%".format(threshold)] = data["input"].progress_map(lambda x : fuzzy_match(x,Y,threshold = threshold))
            return data

    else:
        data = pd.DataFrame(X,columns = ["input"])
        for t in threshold:
            threshold_data = fuzzy_match(X,Y,threshold = t)
            data["match_{}%".format(t)] = threshold_data["match_{}%".format(t)]
        return data









#=============================================================================================================================
# REMOVE FUZZY DUPLICATES
#=============================================================================================================================


def remove_duplicates(texts,distance = None,threshold = 1):
    texts = list(set(texts))
    
    if distance is None:
        distance = levenshtein
        
    new_texts = []
    while len(texts) > 0:
        text = texts[0]
        cut = False
        for t in [t for t in texts if t != text]:
            if distance(text,t) <= threshold:
                cut = True
                break
                
        if not cut:
            new_texts.append(text)
        texts = texts[1:]
        
    return new_texts










#=============================================================================================================================
# DISTANCE
#=============================================================================================================================



def levenshtein(s, t):
    ''' 
    From Wikipedia article; Iterative with two matrix rows.
    Copied from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    '''
    if s == t: return 0
    elif len(s) == 0: return len(t)
    elif len(t) == 0: return len(s)
    v0 = [None] * (len(t) + 1)
    v1 = [None] * (len(t) + 1)
    for i in range(len(v0)):
        v0[i] = i
    for i in range(len(s)):
        v1[0] = i + 1
        for j in range(len(t)):
            cost = 0 if s[i] == t[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]
            
    return v1[len(t)]


    


def levenshtein2(s, t):
    if len(s) < len(t):
        return levenshtein(t, s)

    # So now we have len(s) >= len(t).
    if len(t) == 0:
        return len(s)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    s = np.array(tuple(s))
    t = np.array(tuple(t))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(t.size + 1)
    for s in s:
        # Insertion (t grows longer than s):
        current_row = previous_row + 1

        # Substitution or matching:
        # t and s items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], t != s))

        # Deletion (t grows shorter than s):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]




def jaccard(s,t):
    return jaccard_distance(set(s),set(t))



