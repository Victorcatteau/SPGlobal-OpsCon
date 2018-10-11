#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
NATURAL LANGUAGE PROCESSING 
NAME ENTITY RECOGNITION FUNCTION
Started on the 2018/01/08
theo.alves.da.costa@gmail.com
https://github.com/theolvs

------------------------------------------------------------------------
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk
from collections import defaultdict
import random
from pathlib import Path
import bs4 as bs
import os

# SpaCy
import spacy
from spacy import displacy

# Custom library
from ekimetrics.utils import io


#=============================================================================================================================
# RECOGNIZER
#=============================================================================================================================



class EntityRecognizer(object):
    def __init__(self,model_path = None,model = None):
        """
        NER SpaCy wrapper
        Documentation available at https://spacy.io/usage/training#section-ner
        """
        if model_path is not None:
            self.load_model(model_path)
        elif model is not None:
            self.nlp = model



    #---------------------------------------------------------------------------------
    # IO

    def save_model(self,model_dir,model_name):
        """
        Save a trained model to disk
        :param model_dir: folder path in which to place the saved model folder
        :param model_name: SpaCy model name to save and also name of the folder
        """

        output_dir = Path(os.path.join(model_dir,model_name))
        if not output_dir.exists():
            output_dir.mkdir()
        self.nlp.meta["name"] = model_name
        self.nlp.to_disk(output_dir)
        print("... Saved model to {}".format(output_dir))




    def load_model(self,model_path):
        """
        Load a trained model saved on disk
        :param model_path: folder path in which the model is saved
        """
        print("... Loading model from {}".format(model_path))
        self.nlp = spacy.load(model_path)





    #---------------------------------------------------------------------------------
    # DATA PREPARATION

    def prepare_tagtog_json_data(self,text,json_path):
        """
        Helper functions to convert a json outputted from the annotating website https://tagtog.net
        to the SpaCy training data format
        :param text: input text given also to tagtog.net
        :param json_path: path to the json file exported from tagtog.net
        """

        # Prepare entities, sections and a default dict to store the data
        entities = io.open_json_data(json_path)["entities"]
        sections = [section for section in text.split("\n") if section != ""]
        data = defaultdict(list)

        # Loop over each entity
        for i,entity in enumerate(entities):
            entity_real = entity.get("meta").get("text")
            location = entity.get("meta").get("location")
            label = entity.get("id")
            s,length,section = location.get("start"),location.get("length"),location.get("section")
            e = s + length

            section = sections[section - 1]
            entity_extracted = section[s:e]

            if entity_extracted == entity_real:
                data[section].append((s,e,label))

        # Format to SpaCy format
        for key in data:
            data[key] = {"entities":data[key]}

        return list(data.items())





    #---------------------------------------------------------------------------------
    # TRAINING


    def create_blank(self,base_model = "fr"):
        """
        Prepare blank model for NER training
        :param base_model: the base language on which to build the model
        """

        # Create blank model
        self.nlp = spacy.blank(base_model)

        # Add NER pipe to spacy blank model
        if 'ner' not in self.nlp.pipe_names:
            self.ner = self.nlp.create_pipe('ner')
            self.nlp.add_pipe(self.ner, last=True)

        # otherwise, get it so we can add labels
        else:
            self.ner = self.nlp.get_pipe('ner')




    def train(self,data,n_iter = 100,dropout = 0.5):
        """
        Training the NER model using SpaCy
        :param data: the input data
        :param n_iter: the number of iterations
        :param dropout: make it harder to memorise data
        """
        # Add labels
        for _, annotations in data:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

        # Training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
        with self.nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = self.nlp.begin_training()
            all_losses = []
            for itn in tqdm(range(n_iter)):
                random.shuffle(data)
                losses = {}
                for text, annotations in data:
                    self.nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=dropout,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                all_losses.append(losses["ner"])
        

        # Plot the loss while training
        plt.title("NER training") 
        plt.plot(all_losses)
        plt.show()





    #---------------------------------------------------------------------------------
    # PREDICTION



    def predict(self,text = None,doc = None):
        """
        Apply the NLP and NER model to a text input
        :param text: input text
        :param doc: input document, go through the function untouched
        """
        if doc is None:
            return self.nlp(text)
        else:
            return doc




    def get_entities(self,text = None,doc = None,as_dict = True):
        """
        Get all the entities in a text in a Pandas DataFrame
        :param text: text to be analysed by the NER
        """
        doc = self.predict(text = text,doc = doc)
        df = pd.DataFrame([(ent.text, ent.label_) for ent in doc.ents],columns = ["entity","label"])
        if as_dict:
            d = df.groupby('label')["entity"].apply(list).to_dict()
            return d
        else:
            return df




    #---------------------------------------------------------------------------------
    # VISUALIZATION


    def to_notebook(self,text = None,doc = None):
        """
        Show the entities with DisplaCy on Jupyter notebook
        :param text: text to be analysed by the NER
        """
        doc = self.predict(text = text,doc = doc)
        displacy.render(doc, style='ent',jupyter = True)



    def to_html(self,text = None,doc = None):
        """
        Returns an html snippet for the entity recognizer
        Can be used in Dash applications for example
        :param text: text to be analyzed by the NER
        """

        doc = self.predict(text = text,doc = doc)
        text_html = displacy.render(doc,style = "ent",page = True)
        text_html = bs.BeautifulSoup(text_html,"lxml").find("div")
        return text_html




    def to_dash(self,text = None,doc = None,style = None):
        """
        Returns a dash set of elements for the entity recognizer
        Can be used in Dash applications
        :param text: text to be analyzed by the NER
        """
        import dash_html_components as html
        text_html = self.to_html(text = text,doc = doc)

        def extract_style(el):
            return {k.strip():v.strip() for k,v in [x.split(": ") for x in el.attrs["style"].split(";")]}

        def convert_html_to_dash(el,style = None):
            if type(el) == bs.element.NavigableString:
                return str(el)
            else:
                name = el.name
                style = extract_style(el) if style is None else style
                contents = [convert_html_to_dash(x) for x in el.contents]
                children = []
                for child in contents:
                    if type(child) == str:
                        children.extend([x.strip() if x != "" else html.Br() for x in child.split("\n\n")])
                    else:
                        children.append(child) 
                return getattr(html,name.title())(children,style = style)

        elements = convert_html_to_dash(text_html,style = style)
        return elements


