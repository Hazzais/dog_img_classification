# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:40:52 2019

@author: harry
"""

import argparse
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict the contents of an '
                                                 'image based upon a '
                                                 'pre-trained ResNet50 cnn.')
    parser.add_argument('img_path', type=str,
                        help='The path to the image you want to predict for')
    parser.add_argument('--x-pixels', default=224,
                        help='The number of x pixels to convert the image to')
    parser.add_argument('--y-pixels', default=224,
                        help='The number of y pixels to convert the image to')
    parser.add_argument('-n', '--number-of-predictions', default=10,
                        help='The number of x pixels to convert the image to')

    args = parser.parse_args()

    # Do imports here so that asking for help on the command line does not
    # require the lengthy imports
    # TODO: Optimise the below as it is a bit slow until the model is read in
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing import image
    from keras.applications.resnet50 import preprocess_input
    from keras.applications.resnet50 import decode_predictions

    # TODO: make model choice optional as argument
    model = ResNet50(weights='imagenet')

    # Take image, apply pre-processing, reshape, and predict
    img = image.load_img(args.img_path, target_size=(args.x_pixels,
                                                     args.y_pixels))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions_all = model.predict(x)

    # decode the results into a list of tuples (class, description,
    # probability)
    print('Predicted:', decode_predictions(predictions_all,
                                           top=args.number_of_predictions)[0])
