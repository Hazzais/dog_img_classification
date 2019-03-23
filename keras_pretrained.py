# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:40:52 2019

@author: harry
"""

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'D:\Documents\PythonDoc\Photo_classification\Images\Raw\DSC_1196.JPG'

img = image.load_img(img_path, target_size=(224,224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=10)[0])

