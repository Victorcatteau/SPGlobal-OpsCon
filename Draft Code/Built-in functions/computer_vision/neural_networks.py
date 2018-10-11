#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
COMPUTER VISION
Utils to implement neural networks model
Started on the 30/08/2017

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


# LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


# Keras
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop, Adam
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.models import model_from_json


#=============================================================================================================================
# UTILS 
#=============================================================================================================================


def reload_keras_model(h5_path,json_path):
    from keras.models import model_from_json
    model = model_from_json(open(json_path,"r").read())
    model.load_weights(h5_path)
    return model



def save_keras_model(model,h5_path,json_path):
    model.save(h5_path)
    with open(json_path, "w") as json_file:
        json_file.write(model.to_json())


#=============================================================================================================================
# VANILLA NEURAL NETWORKS
#=============================================================================================================================



def build_binary_vanillaNN_1(input_shape,lr = 0.01):
    model = Sequential()
    model.add(Dense(256,input_shape = input_shape,activation = "relu"))
    # model.add(Dropout(0.25))
    model.add(Dense(256,activation = "relu"))
    # model.add(Dropout(0.25))
    model.add(Dense(1,activation = "sigmoid"))
    optimizer = Adam(lr=lr)
    model.compile(optimizer="sgd", loss='binary_crossentropy',metrics=['accuracy'])
    return model





class VanillaNN(object):
    def __init__(self,input_shape,output_shape,hidden_layers = 2,hidden_layer_size = 256,lr = 0.01,optimizer = "adam",dropout = None):
        input_shape = (input_shape,) if type(input_shape) != tuple else input_shape

        self.model = Sequential()

        # HIDDEN LAYERS
        for layer in range(hidden_layers):
            if layer == 0:
                self.model.add(Dense(hidden_layer_size,input_shape = input_shape,activation = "relu"))
            else:
                self.model.add(Dense(hidden_layer_size,activation = "relu"))

            if dropout is not None:
                self.model.add(Dropout(dropout))


        # FINAL LAYER
        if output_shape > 1:
            final_activation = "softmax"
            loss = "categorical_crossentropy"
        else:
            final_activation = "sigmoid"
            loss = "binary_crossentropy"

        self.model.add(Dense(output_shape,activation = final_activation))


        # OPTIMIZER
        self.model = Adam(lr = lr)

        # COMPILATION
        self.model.compile(optimizer = optimizer,loss = loss,metrics = ["accuracy"])











#=============================================================================================================================
# CONVOLUTIONAL NEURAL NETWORKS
#=============================================================================================================================





def build_binary_CNN_1(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    optimizer = Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])

    return model
