#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
COMPUTER VISION
Utils for computer vision
Started on the 28/12/2016

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


# LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy
import os

# OTHER LIBRARIES
from sklearn.model_selection import train_test_split
from PIL import Image
from io import BytesIO
import ast
import operator
import requests


# OPEN CV
try:
    import cv2
except:
    pass


# KERAS
try:
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input, decode_predictions
except:
    pass



#=============================================================================================================================
# CV MODEL ONTOLOGY
#=============================================================================================================================



class ComputerVisionModel(object):
    def __init__(self,model = None,name = None):

        assert name in [None,"vgg16","vgg19"]

        if model is None:
            self.build_model(name)
        else:
            self.parse_model(model)



    def build_model(self,name):
        pass


    def parse_model(self,model):
        self.model = model







#=============================================================================================================================
# USING PRE TRAINED VGG 16
#=============================================================================================================================



def get_vgg16_model(include_top = True):
    return VGG16(weights='imagenet', include_top=include_top)



def preprocess_image_vgg16(img):
    img = scipy.misc.imresize(img,(224,224),'cubic')
    img = np.asarray(img,dtype = "float32")
    img = np.expand_dims(img,axis = 0)
    img = preprocess_input(img)
    return img



def predict_image_vgg16(img,top = 3,model = None):
    img = np.array(img)
    x = preprocess_image_vgg16(img)
    if model is None:
        model = get_vgg16_model()
    prediction = model.predict(x)
    return decode_predictions(prediction,top = top)






def is_object(image,object_list,threshold = 0.2,value = False,model = None):
    predictions = predict_image_vgg16(image,10,model = model)[0]
    prediction = np.sum([x[2] for x in predictions if x[1] in object_list])
    if value:
        return prediction if prediction > threshold else 0.0
    else:
        return prediction > threshold





def is_bag(image,threshold = 0.2,value = False,model = None):
    bags = ["mailbag","backpack","shopping_basket","purse"]
    return is_object(image,object_list = bags,threshold = threshold,value = value,model = model)













#=============================================================================================================================
# FACE DETECTION
#=============================================================================================================================



def detect_face(img,cascade_classifier = None):
    if cascade_classifier is None:
        cascade_classifier = cv2.CascadeClassifier('C:/git/models/opencv_models/haarcascades/haarcascade_frontalface_default.xml')

    faces = cascade_classifier.detectMultiScale(img, 1.3, 5)

    return faces




def crop_face(img,cascade_classifier = None):
    
    faces = detect_face(img,cascade_classifier = cascade_classifier)

    if len(faces) > 1:
        print("Warning : multiple faces")

    x,y,w,h = faces[0]

    return img[y:y+h, x:x+w]




def draw_face_contours(img,cascade_classifier = None):
    faces = detect_face(img,cascade_classifier = cascade_classifier)

    if len(faces) > 1:
        print("Warning : multiple faces")

    new_img = img.copy()


    for (x,y,w,h) in faces:
        cv2.rectangle(new_img,(x,y),(x+w,y+h),(255,0,0),2)

    return new_img

