#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
OBJECT DETECTION

Started on the 21/08/2017
theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


# Usual libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
import scipy
import os
import os.path
from PIL import Image
from io import StringIO
from collections import defaultdict
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
import json
from tqdm import tqdm

# Tensorflow
import tensorflow as tf

# Ekimetrics Custom Library
from ekimetrics.computer_vision import tf_visualization_utils as vis_util
from ekimetrics.computer_vision import utils






#=============================================================================================================================
# TENSORFLOW OBJECT DETECTION API
#=============================================================================================================================

#------------------------------------------------------------------------------------------------------
# CONSTANTS

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


#------------------------------------------------------------------------------------------------------
# DOWNLOAD MODELS

def download_tf_model(folder = ""):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, os.path.join(folder,MODEL_FILE))
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())



def load_frozen_tf_model(file_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(file_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph







#------------------------------------------------------------------------------------------------------
# CATEGORY MAPPING IO

def save_categories_dictionary(prototext_path,json_path,max_num_classes = 90):
    """
    This function is not applicable here, it needs the utils in the base tensorflow class
    In the file label_map_util.py
    """
    label_map = label_map_util.load_labelmap(prototext_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=max_num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    with open(json_path,"w") as file:
        json.dump(category_index,file)



def load_categories_dictionary(json_path):
    d = json.loads(open(json_path,"r").read())
    for k in d:
        d[int(k)] = d.pop(k)
    return d









#------------------------------------------------------------------------------------------------------
# PREDICTIONS


def predict(detection_graph,categories,images,show = True,save = False,figsize = (12, 8),min_score_thresh=.5,force_cat = None,folder = ""):

    # INIT CACHES
    all_boxes = []
    all_scores = []
    all_classes = []
    all_num_detections = []
    all_raw_images = []
    all_images = []

    # LOAD THE FROZEN COMPUTATIONAL GRAPH
    with detection_graph.as_default():

        # CREATE THE TENSORFLOW SESSION
        with tf.Session(graph=detection_graph) as sess:

            # INPUT MANIPULATION
            if type(images) != list: images = [images]
            image_paths = type(images[0]) == str

            # ITERATE OVER EACH IMAGE
            for image in tqdm(images):

                # Open the image with the PIL library
                if image_paths:
                    if image.startswith("http"):
                        url = image
                        image = utils.open_image_from_url(url)
                    else:
                        image_path = image
                        image = Image.open(image)

                try:
                    all_raw_images.append(image)

                    # Prepare the numpy image
                    image_np = np.array(image.getdata()).reshape((*reversed(image.size), 3)).astype(np.uint8)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    # PREPARE ALL THE TENSORS
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})

                    # Cache the results in the dictionaries
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_classes.append(classes)
                    all_num_detections.append(num_detections)

                    # Remove duplicates on force cat
                    boxes,scores,classes,num_detections = remove_duplicates_on_categories(force_cat,boxes,scores,classes,num_detections)

                    # Adding the bounding boxes
                    vis_util.visualize_boxes_and_labels_on_image_array(image_np,boxes[0],classes[0].astype(np.int32),scores[0],
                                                                       categories,use_normalized_coordinates=True,line_thickness=8,
                                                                       force_cat = force_cat,min_score_thresh = min_score_thresh)


                    # To show the predictions with matplotlib
                    if show:
                        fig = plt.figure(figsize=figsize)
                        plt.imshow(image_np)

                    # To save the images
                    img = Image.fromarray(image_np)

                    all_images.append(img)

                    if save:
                        extension = os.path.splitext(image_path)[1]
                        file_name = image_path.split("/")[-1]
                        img.save(folder + file_name.split(extension)[0]+"_with_boxes"+extension)
                except Exception as e:
                    print(e)


    return all_images,all_raw_images,all_boxes,all_scores,all_classes,all_num_detections





def predict_live(sess,detection_graph,categories,image,show = True,save = False,figsize = (12, 8),min_score_thresh=.5,force_cat = None,folder = ""):

    try:
        # Prepare the numpy image
        image_np = np.array(image.getdata()).reshape((*reversed(image.size), 3)).astype(np.uint8)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # PREPARE ALL THE TENSORS
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})

        # Remove duplicates on force cat
        boxes,scores,classes,num_detections = remove_duplicates_on_categories(force_cat,boxes,scores,classes,num_detections)

        # Adding the bounding boxes
        vis_util.visualize_boxes_and_labels_on_image_array(image_np,boxes[0],classes[0].astype(np.int32),scores[0],
                                                           categories,use_normalized_coordinates=True,line_thickness=8,
                                                           force_cat = force_cat,min_score_thresh = min_score_thresh)

        # To save the images
        img = Image.fromarray(image_np)
    except Exception as e:
        print(e)
        img = image


    return img





def remove_duplicates_on_categories(categories,boxes,scores,classes,num_detections,thresh = 0.1):

    if categories is None:
        return boxes,scores,classes,num_detections
    else:
        b = np.squeeze(boxes)
        c = np.squeeze(classes) 
        s = np.squeeze(scores) 
        df = pd.DataFrame([{"class":c[j],"score":s[j]} for j in range(len(c))]).drop_duplicates(subset = ["class"])
        index = df.loc[df["class"].isin(categories)].index
        
        boxes = np.expand_dims(b[index],axis = 0)
        classes = np.expand_dims(c[index],axis = 0)
        scores = np.expand_dims(s[index],axis = 0)
        num_detections = np.array([len(index)],dtype = "float32")

        return boxes,scores,classes,num_detections






#------------------------------------------------------------------------------------------------------
# PREDICTOR CLASS



class TensorflowDetector(object):
    def __init__(self,model_path,categories_path):
        """
        Helper class to detect object in images

        Arguments: 
            - model_path (string) the complete path to the frozen graph (.pb file)
            - categories_path (string) the complete path to categories dictionary (.json files)
        """

        print(">> Loading tensorflow graph and categories ... ",end = "")
        self.graph = load_frozen_tf_model(model_path)
        self.categories = categories_path
        self.inverse_categories = self.create_inversed_mapping()
        self.boxes = None
        self.scores = None
        self.classes = None
        self.info = None
        self.raw_images = None

        print("ok")


    def create_inversed_mapping(self):
        mapping = {}
        for key in self.categories:
            mapping[self.categories[key]["name"]] = key
        return mapping




    #--------------------------------------------------------------------------------
    # PREDICTION


    def predict(self,images,show = True,save = False,figsize = (12, 8),min_score_thresh=.5,force_cat = None,folder = ""):
        # CONVERT CATEGORIES 
        if force_cat is not None:
            if type(force_cat) != list : force_cat = [force_cat]
            if type(force_cat[0]) != int : force_cat = list(map(lambda x:self.inverse_categories[x],force_cat))

        # PREDICTIONS
        imgs,raw_images,b,s,c,n = predict(self.graph,self.categories,images,show,save,figsize,min_score_thresh,force_cat,folder)

        # CACHING AND ANALYSIS
        self.boxes = b
        self.scores = s
        self.classes = c
        self.info = self.analyze_last_prediction()
        self.raw_images = raw_images
        return imgs






    #--------------------------------------------------------------------------------
    # PREDICTION ANALYSIS



    def analyze_last_prediction(self):
        if self.boxes is None:
            print("Make a prediction first")

        else:
            infos = []
            for i in range(len(self.classes)):
                classes = np.squeeze(self.classes[i]) 
                scores = np.squeeze(self.scores[i]) 
                info = [{"class":classes[j],
                         "category":self.categories[classes[j]]["name"],
                         "score":scores[j]} for j in range(len(classes))]
                infos.append(pd.DataFrame(info,columns = ["class","category","score"]).drop_duplicates(subset = ["category"]))

            return infos




    def get_data(self):
        if self.boxes is None:
            print("Make a prediction first")

        else:
            infos = []
            for i in range(len(self.classes)):
                classes = np.squeeze(self.classes[i]) 
                scores = np.squeeze(self.scores[i])
                boxes = np.squeeze(self.boxes[i])
                info = [{
                         #"class":"ok",
                         "category":self.categories[classes[j]]["name"],
                         # "score":int(np.round(scores[j],3)),
                         # "boxes":list(boxes[j])
                         } for j in range(len(classes))]

                temp = list(set([x["category"] for x in info]))
                info = {"categories":temp}



                infos.append(info)

            return infos





    def is_category(self,category,thresh = 0.05):
        return [category in info["category"].values for info in self.info.loc[self.info["score"] > thresh]]


    def is_person(self,thresh = 0.05):
        return self.is_category("person",thresh)


    def is_handbag(self,thresh = 0.05):
        return self.is_category("handbag",thresh)


    def get_best_object_box(self,category):
        boxes = []
        for i,info in enumerate(self.info):
            info = info.loc[(info["category"] == category)&(info["score"] > 0.05)]
            if len(info) > 0:
                index = info.index[0]
                box = tuple(self.boxes[i][0][index])
            else:
                box = None
            boxes.append(box)
        return boxes


    def crop_best_object_box(self,category):
        boxes = self.get_best_object_box(category)
        cropped_images = []
        for i,img in enumerate(self.raw_images):
            box = boxes[i]
            if box is not None:
                width,height = img.size
                real_box = (box[1]*width,box[0]*height,box[3]*width,box[2]*height)
                cropped_images.append(img.crop(real_box))
            else:
                cropped_images.append(None)

        return cropped_images




    def extract_category(self,category,thresh = 0.05):
        pass







