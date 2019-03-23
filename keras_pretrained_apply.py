# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:48:59 2019

@author: harry
"""

import os
from shutil import copyfile

import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from nltk.corpus import wordnet as wn


dog = wn.synsets('dog')[0]

def get_hyponym(ss):
    out = []
    for ss_i in ss:
        xh = ss_i.hyponyms()
        out += xh
        if len(xh)!=0:
            lt = get_hyponym(xh)
            out += lt

    return out

p = get_hyponym([dog])
dog_words = [synset.name().split('.')[0].lower() for synset in p]


model = ResNet50(weights='imagenet')

#img_path = 'D:\Documents\PythonDoc\Photo_classification\Images\Raw'
img_path = 'D:\Documents\PythonDoc\Photo_classification\Images\Holiday 2015'

#img_name = 'DSC_1196.JPG'

def retrieve_images(folder,image_extensions=('.jpg','.jpeg', '.bmp', '.png', '.gif')):
    base_images = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    return base_images

images_list = retrieve_images(img_path)


def process_images(image_list, folder, x_pixels=224, y_pixels=224, channels=3,\
                   print_status=False):
    images_all_a = np.zeros((len(image_list),x_pixels,y_pixels,channels))
    for i, img_name in enumerate(image_list):
        img_current = image.load_img(os.path.join(img_path,img_name),\
                                     target_size=(x_pixels,y_pixels))
        img_current_a = image.img_to_array(img_current)
        img_current_a = np.expand_dims(img_current_a, axis=0)
        images_all_a[i,:,:,:] = preprocess_input(img_current_a)
        if print_status: print(str(i) + ': ' + img_name)
    return images_all_a

#image_array = process_images(images_list, img_path, x_pixels=224, \
#                             y_pixels=224, channels=3, print_status=True)
#predictions = model.predict(image_array)
#predictions_decoded = decode_predictions(predictions, top=20)
#predictions_dict = dict(zip(images_list, predictions_decoded))
#preds = {}
#for key, val in predictions_dict.items():
#    pct = 0
#    for wrd in val:
#        _, word, prob = wrd
#        is_dog = word.lower() in dog_words
#        if is_dog: pct += prob
#    preds[key] = pct
#
#dog_pictures_list = [key for key, val in preds.items() if val>=0.01]
#no_dog_pictures_list = [key for key, val in preds.items() if val<0.01]


def run_batches(images_list, folder, batch_size, x_pixels=224, y_pixels=224, channels=3):
    n_pictures = len(images_list)
    n_batches = np.ceil(n_pictures/batch_size)

    batch_bounds = [(int(batch_size*(i)), int(np.min([batch_size*(i+1), n_pictures])-1)) \
                    for i in np.arange(0,n_batches)]

    predictions_final = {}
    predictions_all_final = {}

    for i, (low, up) in enumerate(batch_bounds):
        print("Beginning batch: " + str(int(i)) + " of " + str(int(n_batches)) +\
              " --- " + str(low) + " to " + str(up) + " of " +str(n_pictures))
        image_list_batch = images_list[low:up+1]
        image_array_batch = process_images(image_list_batch, folder, x_pixels=x_pixels, \
                             y_pixels=y_pixels, channels=channels, print_status=False)

        predictions_batch = model.predict(image_array_batch)
        predictions_batch_dec = decode_predictions(predictions_batch, top=50)
        predictions_batch_dict = dict(zip(image_list_batch, predictions_batch_dec))

        preds_dog_batch = {}
        for key, val in predictions_batch_dict.items():
            pct = 0
            for wrd in val:
                _, word, prob = wrd
                is_dog = word.lower() in dog_words
                if is_dog: pct += prob
            preds_dog_batch[key] = pct
        predictions_final.update(preds_dog_batch)
        predictions_all_final.update(predictions_batch_dict)

    return predictions_final, predictions_all_final


dog_predictions, preds = run_batches(images_list, img_path, 250,\
                              x_pixels=224, y_pixels=224, channels=3)



dog_pictures_list = [key for key, val in dog_predictions.items() if val>=0.01]
no_dog_pictures_list = [key for key, val in dog_predictions.items() if val<0.01]

folder_true = r'D:\Documents\PythonDoc\Photo_classification\Images\Classified\Pretrained_RESNET\Dog-holiday'

folder_false = r'D:\Documents\PythonDoc\Photo_classification\Images\Classified\Pretrained_RESNET\NoDog-holiday'


for img in dog_pictures_list:
    _=copyfile(os.path.join(img_path,img), os.path.join(folder_true,img))

for img in no_dog_pictures_list:
    _=copyfile(os.path.join(img_path,img), os.path.join(folder_false,img))



