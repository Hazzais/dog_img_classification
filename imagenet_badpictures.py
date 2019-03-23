# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 19:05:35 2019

@author: harry
"""

import os
import skimage
from skimage import io, transform
import numpy as np

folder = 'D:\Documents\PythonDoc\Photo_classification\Images\Other_sources\ImageNet\Dogs'

picture_list = ['2.jpg',
                '5.jpg',
                '6.jpg',
                '17.jpg',
                '18.jpg',
                '28.jpg',
                '34.jpg',]

f = io.imread(os.path.join(folder,picture_list[0]))

f2 = io.imread(os.path.join(folder,picture_list[6]))

same = np.array_equal(f,f2)
