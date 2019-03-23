# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:56:04 2019

@author: harry
"""

import glob
import os
import re
from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import time
import matplotlib.pyplot as plt


print('GPU available for TensorFlow: '+str(tf.test.is_gpu_available(cuda_only=True)))

image_path_raw = 'D:\Documents\PythonDoc\Photo_classification\Images\Raw'
image_path_processed = 'D:\Documents\PythonDoc\Photo_classification\Images\Processed'
image_path_caltech256 = r'D:\Documents\PythonDoc\Photo_classification\Images\Caltech256\256_ObjectCategories'

x_pixels = 64
y_pixels = 64


def retrieve_images(folder,image_extensions=('.jpg','.jpeg', '.bmp', '.png', '.gif')):
    base_images = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    return base_images


def process_images(image_list,folder,x_pixels=64,y_pixels=64,test_num=None,print_status=False):

    processed_images = np.zeros((len(image_list),x_pixels*y_pixels))
    image_ids = {}
    for img_num, img_name in enumerate(image_list):
        if isinstance(test_num,int) and img_num>test_num:
            break

        image_ids[img_name]=img_num
        fname = os.path.join(folder,img_name)
        if print_status: print(str(img_num)+': '+img_name)
        img = cv2.imread(fname,0)
        img = cv2.resize(img,(x_pixels,y_pixels),
                       fx=0,
                       fy=0)
        img_flat = img.flatten().transpose()
        processed_images[img_num,:] = img_flat

    return processed_images

def show_picture(image_array,image_labels=None,number=0,cmap='seismic',x_pixels=64,y_pixels=64):
    if image_labels!=None:
        print('Picture label: ' + image_labels[number])
    test=np.reshape(image_array[number,:],(x_pixels,y_pixels))
    io.imshow(test,cmap=cmap)


hf_images = retrieve_images(image_path_raw)
hf_images_processed = process_images(hf_images,image_path_raw,x_pixels=x_pixels,y_pixels=y_pixels)






def caltech256_images(folder_base,test_num=None):

    subfolders = [f.path for f in os.scandir(folder_base) if f.is_dir()]
    all_ct256_images = np.array([]).reshape(0,x_pixels*y_pixels)
    all_ct256_labels = []

    for i, folder in enumerate(subfolders):
        if isinstance(test_num,int) and i>test_num:
            break
        temp = subfolders[i].rsplit('\\', 1)[1]
        image_label = temp[temp.rfind('.')+1:]
        print('For folder ' + str(i+1) + ' of ' + str(len(subfolders)) + ': --- ' + image_label)
        base_images = retrieve_images(folder)
        base_images_processed = process_images(base_images,folder,x_pixels=x_pixels,y_pixels=y_pixels)
        labels = [image_label for a in range(len(base_images_processed))]

        # Eventually slow as involves copying
        all_ct256_images = np.vstack((all_ct256_images,base_images_processed))
        all_ct256_labels = all_ct256_labels + labels

    return (all_ct256_labels,all_ct256_images)

all_ct256_labels, all_ct256_images = caltech256_images(image_path_caltech256,test_num=None)

unique_categories = list(set(all_ct256_labels))


all_images = all_ct256_images

pixel_mean = np.mean(all_images, axis=0)
pixel_std = np.std(all_images, axis=0)

all_images_norm = (all_images-pixel_mean)/(pixel_std+1e-8)

show_picture(all_ct256_images,image_labels=all_ct256_labels,number=3460,cmap='gray')





plt.figure()
show_picture(all_images_norm,image_labels=all_ct256_labels,number=0,cmap='gray')




#
## CIFAR images
#def unpickle(file):
#    import pickle
#    with open(file, 'rb') as fo:
#        dict = pickle.load(fo, encoding='bytes')
#    return dict
#
#imloc = 'D:\Documents\PythonDoc\Photo_classification\Images\Other_sources\cifar-10-batches'
#batch1 = unpickle(os.path.join(imloc,'data_batch_1'))
#batch2 = unpickle(os.path.join(imloc,'data_batch_2'))
#batch3 = unpickle(os.path.join(imloc,'data_batch_3'))
#batch4 = unpickle(os.path.join(imloc,'data_batch_4'))
#batch5 = unpickle(os.path.join(imloc,'data_batch_5'))
#batch_test = unpickle(os.path.join(imloc,'test_batch'))
#
#
#cifar_imgs = np.array(batch1[b'data'])
#
#df = pd.DataFrame(batch1[b'data'])
#df['image'] = df.values.tolist()
#df.drop(range(3072),axis=1,inplace=True)
#df['label'] = batch1[b'labels']
#
#
#def get_dataframe(batch):
#    df = pd.DataFrame(batch[b'data'])
#    df['image'] = df.values.tolist()
#    df.drop(range(3072),axis=1,inplace=True)
#    df['label'] = batch[b'labels']
#    return df
#
#train = pd.concat([get_dataframe(batch1),
#                   get_dataframe(batch2),
#                   get_dataframe(batch3),
#                   get_dataframe(batch4),
#                   get_dataframe(batch5)],
#        ignore_index=True)
#test = get_dataframe(batch_test)
