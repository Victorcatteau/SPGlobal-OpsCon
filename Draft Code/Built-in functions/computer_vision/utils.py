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
import os
import glob

# OTHER LIBRARIES
from sklearn.model_selection import train_test_split
import ast
import operator
from PIL import Image
import requests
from io import BytesIO


import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont


try:
    import cv2
except:
    pass





#=============================================================================================================================
# IO
#=============================================================================================================================





def open_image_from_url(url):
    r = requests.get(url)
    img = Image.open(BytesIO(r.content),"r")
    return img




def download_image_from_url(url,path):
    """
    IO function that downloads a picture from its url
    Takes as input : 
        - The url
        - The path for the image
    Returns: None
    """
    img = open_image_from_url(url)
    img.save(path)








def build_dataset_from_folder(folder,colors = False,resize = (0,0),max_images = None,shuffle = False,return_list = False):
    """
    IO function to build a dataset from the path of a folder containing images
    Takes as input : 
        - The path for the folder
        - a boolean for colors if you want colored images or black&white
        - a tuple for resize to resize or not every images in the folder
        - max_images (None by default) to only read the top images
        - shuffle a boolean to read them in a random order

    Returns a 3 or 4 dimensional numpy tensor (number of images,length,width,depth (if colors = True))
    """

    X = []
    list_of_images = glob.glob(folder+"*")

    if shuffle:
        np.random.shuffle(list_of_images)
        
    for i,image in enumerate(list_of_images):
        if max_images is not None:
            if i >= max_images:
                break 
        try:
            print("\rReading from folder %s : [%s/%s]"%(folder,i+1,len(list_of_images)),end = "")
            img = read_image_from_file(image,colors = colors,resize = resize)
            img = np.expand_dims(np.array(img),axis = 0)
            if i == 0:
                X = img
            else:
                X = np.vstack([X,img])
        except OSError:
            pass
            
    print('')

    if not return_list:
        return X
    else:
        return X,list_of_images







def extract_ok_urls(urls):
    """
    Helper function that takes a list of images url, tests them if available (HTTP response 200)
    Returns: a list of available picture urls
    """

    return [url for url in urls if requests.get(url).ok] 






def read_image_from_url(url,colors = False,resize = None):
    """
    IO function that reads a picture from an url
    Takes as input : 
        - The url
        - colors = True returns a 3 dimensional array, False a two dimensional array for the picture in black & white
        - resize a tuple to resize the picture at a given pixels shape
    Returns: a numpy array for the pixel value of the image
    """
    request = requests.get(url)
    if request.ok:
        img = scipy.misc.imread(BytesIO(request.content),flatten = not colors)
        if resize is not None:
            img = scipy.misc.imresize(img,resize,'cubic')
        return img

    else:
        print("Invalid request")




def read_image_from_file(file_path,colors = True,resize = None):
    """
    IO function that reads a picture from its path
    Takes as input : 
        - The file path
        - colors = True returns a 3 dimensional array, False a two dimensional array for the picture in black & white
        - resize a tuple to resize the picture at a given pixels shape
    Returns: a PIL Image
    """
    
    # OPEN AN IMAGE
    img = Image.open(file_path)

    # CONVERT TO COLORS OR GRAYSCALE
    img = img.convert("RGB" if colors else "L")

    # RESIZE THE IMAGE
    if resize is not None:
        img = img.resize(resize)

    return img










#=============================================================================================================================
# PLOTTING IN NOTEBOOKS
#   Functions helpful to plot images in a Jupyter notebook
#=============================================================================================================================


def plot_gallery(images, titles,n_row=3, n_col=6):
    #Helper function to plot a gallery of images
    plt.figure(figsize=(2.3 * n_col, 2.3 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i],interpolation="nearest")
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def plot_image(array,title = ""):
    plt.title(title)
    plt.imshow(array,interpolation='nearest')
    plt.show()


def plot_list_of_urls(urls):
    for url in urls:
        image = read_img(url)
        plt.figure()
        plt.imshow(image)
        plt.show()









#=============================================================================================================================
# OBJECT DETECTION
#=============================================================================================================================



def crop_image(image,box):
    """
    :Image is a PIL image
    :Box should be a 4 tuple of ymin,xmin,ymax,xmax
    returns: the cropped image
    """
    return image.crop(box)





def get_best_bounding_box_with_sliding_window(img, predict_fn, step=10, window_sizes=[20,40,60,80,100,120,140,160]):
    best_box = None
    # best_box_prob = -np.inf
    best_box_prob = 0.0
    img = np.array(img)

    # loop window sizes: 20x20, 30x30, 40x40...160x160
    def get_range(size,axis):
        return range(0, img.shape[axis] - size + 1, step)

    ys = get_range(window_sizes[0],axis = 0)
    xs = get_range(window_sizes[0],axis = 1)
    total = len(window_sizes)*len(xs)*len(ys)
    i = 0
    
    # Window size iterations
    for window_size in window_sizes:
        for ymin in get_range(window_size,axis = 0):
            for xmin in get_range(window_size,axis = 1):

                xmax,ymax = xmin + window_size,ymin + window_size
                box = (ymin,xmin,ymax,xmax)

                # crop the original image
                cropped_img = img[ymin:ymax, xmin:xmax]

                box_prob = predict_fn(cropped_img)
                print('\r[{}/{}] Predicted for box {} - {}'.format(i+1,total,box,box_prob),end = "")
                if box_prob > best_box_prob:
                    best_box = box
                    best_box_prob = box_prob
                    
                i += 1

    print()

    return best_box







#=============================================================================================================================
# DRAWING ON IMAGES
# Taken from Tensorflow developers
# https://github.com/tensorflow/models/blob/master/object_detection/utils/visualization_utils.py
#=============================================================================================================================





def draw_bounding_boxes_on_image(image,boxes,color='red',thickness=4,display_str_list_list=()):
    """Draws bounding boxes on image.
    Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (xmin, ymin, xmax, ymax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.
    Raises:
    ValueError: if boxes is not a [N, 4] array
    """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],boxes[i, 3], color, thickness, display_str_list)




def draw_bounding_box_on_image(image,box,color='red',thickness=4,display_str_list=(),use_normalized_coordinates=False):
    """Adds a bounding box to an image.
    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
    """
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)
    im_width, im_height = image.size
    ymin,xmin,ymax,xmax = box

    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    draw.line([(left, top), (left, bottom), (right, bottom),(right, top), (left, top)], width=thickness, fill=color)
    
    try:
        font = ImageFont.truetype('arial.ttf', 16)
    except IOError:
        font = ImageFont.load_default()

    text_bottom = top

    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),display_str,fill='black',font=font)
        text_bottom -= text_height - 2 * margin

    return new_image



