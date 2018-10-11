#!/usr/bin/env python
# -*- coding: utf-8 -*- 



"""--------------------------------------------------------------------
PREPROCESSING
Started on the 30/10/2017


https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0
theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import numpy as np
import cv2
from PIL import Image




#=============================================================================================================================
# HELPER FUNCTIONS
#=============================================================================================================================


def to_black_and_white(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def detect_edges(img,threshold1 = 200,threshold2 = 300):
    return cv2.Canny(img, threshold1 = threshold1, threshold2=threshold2)


def gaussian_smooth(img):
    return cv2.GaussianBlur(img,(5,5),0)


def select_part_from_mask(img,vertices):
    vertices = np.array([vertices],dtype = np.int32)
    
    mask = np.zeros_like(img)
    
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked






#=============================================================================================================================
# BRIGHTNESS ANALYSIS
# 
#   Algorithms utils to study the brightness of an image
#=============================================================================================================================



def brightness_dim_2(image):
    """
    This function takes as input a image stored in a numpy array of shape (height,width) (2 dimensional)
    Returns the mean brightness : 100% is full white 0% is black
    """

    return(np.mean(255-image))/255







def brightness_dim_3(image):
    """
    This function takes as input a image stored in a numpy array of shape (height,width,color depth) (3 dimensional)
    Returns the mean brightness : 100% is full white 0% is black
    """

    return 1-((np.mean(255-image))/255)







def display_image_sorted_by_brightness(urls):
    """
    Helper function to sort a list of urls of images by brightness
    Deprecated
    """

    brightness ={}
    for filename in urls : 
        image = read_img(filename)
        if len(image.shape)>2:
            brightness[filename] = brightness_dim_3(image)
        else : 
            brightness[filename] = brightness_dim_2(image)

    list_sorted = sorted(brightness.items() , key = operator.itemgetter(1) , reverse=True)
    return plot_list_of_urls([x[0] for x in list_sorted])












#=============================================================================================================================
# CLARITY ANALYSIS
#=============================================================================================================================



def display_image_sorted_by_clarity(urls):
    """
    Helper function to sort a list of urls of images by clarity (is my image blurry ?)
    Uses the library OpenCV
    Deprecated
    """

    all_ratio = {}
    for filename in urls:
        orig = read_img(filename)
        sobel_dx = cv2.Sobel(orig, cv2.CV_64F, 1, 0, ksize=5)
        sobel_dy = cv2.Sobel(orig, cv2.CV_64F, 0, 1, ksize=5)
        magnitude_image = cv2.magnitude(sobel_dx,sobel_dy,sobel_dx)
        mag, ang = cv2.cartToPolar(sobel_dx, sobel_dy, magnitude_image) 
        ratio = cv2.sumElems(mag[0])
        all_ratio[filename] = ratio[0]

    list_sorted = sorted(all_ratio.items(), key=operator.itemgetter(1) , reverse=True)
    return plot_list_of_urls([x[0] for x in list_sorted])












#=============================================================================================================================
# DOMINANT COLOR ANALYSIS
#=============================================================================================================================



def blue_image(image):
    """
    Study the % of pixels where the dominant color is blue
    Can be used to easily detect pictures of the sea or the sky for example
    
    Takes as input a numpy array representing the picture
    Returns: the coefficient for blue dominance
    """

    if len(image.shape) <=2:
        return None
    else:
        number_of_pixels = image.shape[0]*image.shape[1]
        image_reshape = np.reshape(image,(number_of_pixels,3))
        number_of_blue_pixels = len([x for x in image_reshape if x[2]>x[1] and x[2]>x[0]])
        return (number_of_blue_pixels/number_of_pixels)






def sorted_blue_image(urls):
    """
    Helper function to sort a list of urls of images by blue dominance (is my image blue ?)
    Deprecated
    """
    list_image ={}
    for filename in urls : 
        image = read_img(filename)
        list_image[filename] = blue_image(image)
    
    list_sorted = sorted(list_image.items(), key=operator.itemgetter(1) , reverse=True)
    return plot_list_of_urls([x[0] for x in list_sorted])







