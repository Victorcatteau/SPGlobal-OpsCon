#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
SENTIMENT ANALYSIS
Started on the 16/10/2017

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


# Usual
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import json
from tqdm import tqdm
import emoji


# Textblob
try:
    from textblob import TextBlob
except:
    pass


# NLTK
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer # nltk v3.2.4+
except:
    pass


# TorchMoji
try:
    from torchmoji.sentence_tokenizer import SentenceTokenizer
    from torchmoji.model_def import torchmoji_emojis
    from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
except:
    pass



#=============================================================================================================================
# TORCHMOJI PREDICTOR
#=============================================================================================================================


INDEX_EMOJI = {
    0:":joy:",
    1:":unamused:",
    2:":weary:",
    3:":sob:",
    4:":heart_eyes:",
    5:":pensive:",
    6:":ok_hand:",
    7:":blush:",
    8:":heart:",
    9:":smirk:",
    10:":grin:",
    11:":notes:",
    12:":flushed:",
    13:":100:",
    14:":sleeping:",
    15:":relieved:",
    16:":relaxed:",
    17:":raised_hands:",
    18:":two_hearts:",
    19:":expressionless:",
    20:":sweat_smile:",
    21:":pray:",
    22:":confused:",
    23:":kissing_heart:",
    24:":heart:",
    25:":neutral_face:",
    26:":information_desk_person:",
    27:":disappointed:",
    28:":see_no_evil:",
    29:":weary:",
    30:":v:",
    31:":sunglasses:",
    32:":rage:",
    33:":+1:",
    34:":cry:",
    35:":sleepy:",
    36:":yum:",
    37:":triumph:",
    38:":hand:",
    39:":mask:",
    40:":clap:",
    41:":eyes:",
    42:":gun:",
    43:":persevere:",
    44:":smiling_imp:",
    45:":disappointed_relieved:",
    46:":broken_heart:",
    47:":heartpulse:",
    48:":musical_note:",
    49:":speak_no_evil:",
    50:":wink:",
    51:":skull:",
    52:":confounded:",
    53:":smile:",
    54:":stuck_out_tongue_winking_eye:",
    55:":angry:",
    56:":no_good:",
    57:":muscle:",
    58:":facepunch:",
    59:":purple_heart:",
    60:":sparkling_heart:",
    61:":blue_heart:",
    62:":grimacing:",
    63:":sparkles:",
}



def _top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]






class TorchMojiPredictor(object):
    def __init__(self,model_path = PRETRAINED_PATH,vocab_path = VOCAB_PATH,max_length = 30):

        # Load arguments
        self.max_length = max_length
        self.model_path = model_path
        self.vocab_path = vocab_path

        # Load model and vocabulary
        self.build_model(self.model_path)
        self.build_vocabulary(self.vocab_path,self.max_length)



    #--------------------------------------------------------
    # LOADING

    def build_model(self,path):
        self.model = torchmoji_emojis(PRETRAINED_PATH)


    def build_vocabulary(self,path,max_length = 30):
        print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)

        self.tokenizer = SentenceTokenizer(vocabulary, self.max_length)


    #--------------------------------------------------------
    # TOKENIZER





    def tokenize(self,texts):
        if type(texts) != list: texts = [texts]
        tokenized, _, _ = self.tokenizer.tokenize_sentences(texts)
        return tokenized


    def predict_proba(self,texts):
        probas = self.model(self.tokenize(texts))
        return probas


    def predict_sentiment(self,texts,top = 5):
        if type(texts) != list: texts = [texts]
        probas = self.predict_proba(texts)
        sentiments = [DeepMojiSentiment(text,proba,top = 5) for text,proba in zip(texts,probas)]
        return sentiments






class DeepMojiSentiment(object):
    def __init__(self,text,probas,top = 5):
        self.text = text
        self.probas = probas
        self.top = top
        self.emotions = self.get_top_emotions(self.top)


    def __repr__(self):
        return self.text + " : " + self._clean_emotions()


    def __str__(self):
        return self.__repr__()



    def get_top_emotions(self,top = 5):
        top_emotions = _top_elements(self.probas,top)
        return [(index,proba) for index,proba in zip(top_emotions,self.probas[top_emotions])]


    def _clean_emotions(self):
        return " - ".join(["{} {}%".format(emoji.emojize(INDEX_EMOJI[index],use_aliases = True),round(proba*100,1)) for index,proba in self.emotions])




