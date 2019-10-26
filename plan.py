# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 19:34:51 2019

@author: harry
"""

"""

# Image collection
 Dogs:
 - ImageNet dog pictures
 - Stanford dogs dataset
 No dogs:
 - UKBench
 - Caltech non-dog pictures
 - ImageNet non-dog pictures?

 Aim for training set of about 10,000 to begin with

# Preprocessing
 - Remove bad pictures
 - Augment pictures
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
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import gc

img_path_dogs ='D:\Documents\PythonDoc\Photo_classification\Images\StanfordDogs'
img_path_caltech = r'D:\Documents\PythonDoc\Photo_classification\Images\Caltech256\256_ObjectCategories'
x_pixels = 32
y_pixels = 32
color = False

def retrieve_images(folder,image_extensions=('.jpg','.jpeg', '.bmp', '.png', '.gif')):
    base_images = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    return base_images


def process_images(image_list, folder, x_pixels=224, y_pixels=224, color=True,\
                   print_status=False):
    channels = 3 if color else 1
    gs_bool = False if color else True

    images_all_a = np.zeros((len(image_list),x_pixels,y_pixels,channels))

    for i, img_name in enumerate(image_list):
        img_current = image.load_img(os.path.join(folder,img_name),\
                                     target_size=(x_pixels,y_pixels),
                                     grayscale=gs_bool)
        img_current_a = image.img_to_array(img_current)
        images_all_a[i,:,:,:] = np.expand_dims(img_current_a, axis=0)

        if print_status: print(str(i) + ': ' + img_name)
    return images_all_a



def load_caltech(folder_base, x_pixels=64, y_pixels=64, color=True, test_num=None):
    channels = 3 if color else 1

    subfolders = [f.path for f in os.scandir(folder_base) if f.is_dir()]

    n_images = 0
    image_count = {}
    # Find number of images for array to be set
    for i, folder in enumerate(subfolders):
        temp = subfolders[i].rsplit('\\', 1)[1]
        image_label = temp[temp.rfind('.')+1:]
        base_images = retrieve_images(folder)
        n_images += len(base_images)
        image_count[image_label] = len(base_images)

    all_ct256_images = np.zeros((n_images, x_pixels, y_pixels, channels))

    print(all_ct256_images.shape)

    all_ct256_labels = []

    min_bound = 0
    max_bound = 0
    for i, folder in enumerate(subfolders):
        if isinstance(test_num,int) and i>test_num:
            break
        temp = subfolders[i].rsplit('\\', 1)[1]
        image_label = temp[temp.rfind('.')+1:]
        print('For folder ' + str(i+1) + ' of ' + str(len(subfolders)) + ': --- ' + image_label)
        base_images = retrieve_images(folder)
        max_bound = min_bound + image_count[image_label]
        all_ct256_images[min_bound:max_bound,:,:,:] = process_images(base_images,
                        folder, x_pixels=x_pixels, y_pixels=y_pixels,
                        print_status=False, color=color)

        min_bound = max_bound
        all_ct256_labels = all_ct256_labels + [image_label] * image_count[image_label]

    return all_ct256_images, all_ct256_labels



def load_Stanforddogs(folder_base, x_pixels=64, y_pixels=64, color=True, test_num=None):
    channels = 3 if color else 1

    subfolders = [f.path for f in os.scandir(folder_base) if f.is_dir()]

    n_images = 0
    image_count = {}
    # Find number of images for array to be set
    for i, folder in enumerate(subfolders):
        temp = subfolders[i].rsplit('\\', 1)[1]
        image_label = temp[temp.find('-')+1:]
        base_images = retrieve_images(folder)
        n_images += len(base_images)
        image_count[image_label] = len(base_images)

    all_dog_images = np.zeros((n_images, x_pixels, y_pixels, channels))

    print(all_dog_images.shape)

    all_dog_labels = []

    min_bound = 0
    max_bound = 0
    for i, folder in enumerate(subfolders):
        if isinstance(test_num,int) and i>test_num:
            break
        temp = subfolders[i].rsplit('\\', 1)[1]
        image_label = temp[temp.find('-')+1:]
        print('For folder ' + str(i+1) + ' of ' + str(len(subfolders)) + ': --- ' + image_label)
        base_images = retrieve_images(folder)
        max_bound = min_bound + image_count[image_label]
        all_dog_images[min_bound:max_bound,:,:,:] = process_images(base_images,
                        folder, x_pixels=x_pixels, y_pixels=y_pixels,
                        print_status=False, color=color)

        min_bound = max_bound
        all_dog_labels = all_dog_labels + [image_label] * image_count[image_label]

    return all_dog_images, all_dog_labels


clt_images, clt_labels = load_caltech(img_path_caltech, \
                                      x_pixels=x_pixels, y_pixels=y_pixels, color=color, test_num=None)

sfd_images, sfd_labels = load_Stanforddogs(img_path_dogs, x_pixels=x_pixels,
                                           y_pixels=y_pixels, color=color, test_num=None)


def show_picture(image_array,image_labels=None,number=0,cmap='seismic',x_pixels=64,y_pixels=64):
    if image_labels!=None:
        print('Picture label: ' + image_labels[number])
    test=np.reshape(image_array[number,:],(x_pixels,y_pixels))
    io.imshow(test,cmap=cmap)


def show_keras_array_picture(image_array, labels=None, number=0):
    plt.figure()
    if labels!=None:
        plt.title('Picture:' + labels[number])
    img = image_array[number,:,:,:]
    dim3 = img.shape[2]
    if dim3==1:
        img2 = np.squeeze(img)
        plt.imshow(img2, cmap='gray')
    else:
        plt.imshow(img/255)


#Labrador_retriever
show_keras_array_picture(clt_images, clt_labels, number=8888)
show_keras_array_picture(sfd_images, sfd_labels, number=18643)

pictures_clt = np.array([0 if p!='dog' else 1 for p in clt_labels])
pictures_sfd = np.array([1 for p in sfd_labels])

all_pictures = np.concatenate((clt_images, sfd_images))
all_labels = np.concatenate((pictures_clt, pictures_sfd))


del clt_images, clt_labels, sfd_images, sfd_labels
gc.collect()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
b = OneHotEncoder(sparse=False)
all_labels_binary = b.fit_transform(all_labels.reshape(-1,1))


X_train, X_test, y_train, y_test = train_test_split(all_pictures, all_labels_binary,
                                                    test_size=0.33, random_state=42)
X_train = X_train/255
X_test = X_test/255

train_base_folder = 'D:\Documents\PythonDoc\Photo_classification\Images\Processed\test'
test_base_folder = 'D:\Documents\PythonDoc\Photo_classification\Images\Processed\test'




img_gen = image.ImageDataGenerator(featurewise_std_normalization=True,
                                   zca_whitening=True,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True)

img_gen.fit(X_train)

# https://www.kaggle.com/mgmarques/cnn-exercise-deep-learning-for-computer-vision
num_classes = 2
input_shape = (64, 64, 1)
kernel = (3, 3)
seed = 42
np.random.seed(seed)

from keras.optimizers import Adam
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

#model = Sequential()
#model.add(Conv2D(64, kernel_size=kernel, activation='relu', input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(128, kernel_size=kernel, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
##model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(64,64,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

#SVG(model_to_dot(model, show_shapes=True, show_layer_names=True, rankdir='TB').create(prog='dot', format='svg'))

batch_size = 50
epochs = 5
lrate = 0.1
epsilon = 1e-8
decay = 1e-4

optimizer = Adam(lr=lrate)

model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)


img_idx = 122
test_image = np.expand_dims(X_test[img_idx,:,:,0], axis=0)
x = model.predict_classes(test_image,batch_size=1)



plt.imshow(X_test[img_idx,:,:,0],aspect='auto')
print('Actual label:', labelNames[np.argmax(y_test[img_idx,:,:,1])])
# Preper image to predict
test_image =np.expand_dims(X_test[img_idx], axis=0)
print('Input image shape:',test_image.shape)
print('Predict Label:',labelNames[model.predict_classes(test_image,batch_size=1)[0]])
print('\nPredict Probability:\n', model.predict_proba(test_image,batch_size=1))