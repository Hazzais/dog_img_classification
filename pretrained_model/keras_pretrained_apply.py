# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:48:59 2019

@author: harry
"""

import os
import sys
from shutil import copyfile

import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from nltk.corpus import wordnet as wn


def get_hyponym(ss):
    """ Recursively get hyponyms of a given word from a WordNet synset   """
    out = []
    for ss_i in ss:
        xh = ss_i.hyponyms()
        out += xh
        if len(xh) != 0:
            lt = get_hyponym(xh)
            out += lt
    return out


def retrieve_images(folder,
                    image_extensions=('.jpg',
                                      '.jpeg',
                                      '.bmp',
                                      '.png',
                                      '.gif')):
    """ Retrieve images from a given folder  """
    base_images = [f for f in os.listdir(folder)
                   if f.lower().endswith(image_extensions)]
    return base_images


def process_images(image_list, img_path, x_pixels=224, y_pixels=224,
                   channels=3, print_status=False):
    """ Pre-process images so that the pretrained model can be used to predict
    their contents  """
    images_all_a = np.zeros((len(image_list), x_pixels, y_pixels, channels))
    for i, img_name in enumerate(image_list):
        img_current = image.load_img(os.path.join(img_path, img_name),
                                     target_size=(x_pixels, y_pixels))
        img_current_a = image.img_to_array(img_current)
        img_current_a = np.expand_dims(img_current_a, axis=0)
        images_all_a[i, :, :, :] = preprocess_input(img_current_a)
        if print_status:
            print(str(i) + ': ' + img_name)
    return images_all_a


def run_batches(images_list, img_path, batch_size, x_pixels=224, y_pixels=224,
                channels=3, top_n_preds=50):
    """ Taking a batch of images, pre-process and then predict contents """
    n_pictures = len(images_list)
    n_batches = np.ceil(n_pictures/batch_size)
    batch_bounds = [(int(batch_size*i),
                     int(np.min([batch_size*(i+1), n_pictures])-1))
                    for i in np.arange(0, n_batches)]

    predictions_final = {}
    predictions_all_final = {}

    for i, (low, up) in enumerate(batch_bounds):
        print("Beginning batch: " + str(int(i)) + " of " + str(int(n_batches))
              + " --- " + str(low) + " to " + str(up) + " of "
              + str(n_pictures))
        image_list_batch = images_list[low:up+1]
        image_array_batch = process_images(image_list_batch,
                                           img_path,
                                           x_pixels=x_pixels,
                                           y_pixels=y_pixels,
                                           channels=channels,
                                           print_status=False)

        # Make predictions and decode
        predictions_batch = model.predict(image_array_batch)
        predictions_batch_dec = decode_predictions(predictions_batch,
                                                   top=top_n_preds)
        predictions_batch_dict = dict(
            zip(image_list_batch, predictions_batch_dec))

        # Check whether top words are in the list of related words (hyponyms)
        # for requested word - sum probability to get final predicted
        # probability
        preds_image_batch = {}
        for key, val in predictions_batch_dict.items():
            pct = 0
            for wrd in val:
                _, word, prob = wrd
                is_image = word.lower() in image_words
                if is_image:
                    pct += prob
            preds_image_batch[key] = pct
        predictions_final.update(preds_image_batch)
        predictions_all_final.update(predictions_batch_dict)

    return predictions_final, predictions_all_final


if __name__ == '__main__':

    # TODO: convert the following to command line args
    img_path = 'D:\Documents\PythonDoc\Photo_classification\Images\Holiday 2015'
    folder_true = r'D:\Documents\PythonDoc\Photo_classification\Images\Classified\Pretrained_RESNET\Dog-holiday'
    folder_false = r'D:\Documents\PythonDoc\Photo_classification\Images\Classified\Pretrained_RESNET\NoDog-holiday'
    word_of_interest = 'dog'
    xpixels = 224
    ypixels = 224
    channels = 3
    top_n_preds = 50
    verbose = False
    batch_size = 250
    prediction_threshold = 0.01

    # Get all words in ImageNet (using WordNet) corresponding to chosen word
    try:
        synset_of_interest = wn.synsets('fgd')[0]
    except IndexError as e:
        print('Chosen word {} may not exist in WordNet'
              .format(word_of_interest))
        sys.exit(1)

    p = get_hyponym([synset_of_interest])
    image_words = [synset.name().split('.')[0].lower() for synset in p]

    # Retrieve final weights of imagenet network
    model = ResNet50(weights='imagenet')

    # Retrieve images from specified input folder
    images_list = retrieve_images(img_path)

    # Make predictions
    image_predictions, predictions_long = run_batches(images_list,
                                                      img_path,
                                                      batch_size,
                                                      x_pixels=xpixels,
                                                      y_pixels=ypixels,
                                                      channels=channels,
                                                      top_n_preds=top_n_preds)

    # Create two lists, one for images containing the item specified,
    # and another for those which do not
    image_pictures_list = [key for key, val in image_predictions.items()
                           if val >= prediction_threshold]
    no_image_pictures_list = [key for key, val in image_predictions.items()
                              if val < prediction_threshold]

    # Copy into separate folders
    # TODO: might want to rethink this and create some sort of data structure
    #  etc.
    for img in image_pictures_list:
        _ = copyfile(os.path.join(img_path, img),
                     os.path.join(folder_true, img))

    for img in no_image_pictures_list:
        _ = copyfile(os.path.join(img_path, img),
                     os.path.join(folder_false, img))
