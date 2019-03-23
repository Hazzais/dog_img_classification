# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 19:16:35 2019

@author: harry
"""

from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import urllib
import os

page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02099712")

soup = BeautifulSoup(page.content, 'html.parser')

str_soup=str(soup)#convert soup to string so it can be split

split_urls=str_soup.split('\r\n')#split so each url is a different possition on a list

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


image_base_folder = 'D:\Documents\PythonDoc\Photo_classification\Images\Other_sources\ImageNet\Dogs'

img_rows, img_cols = 64, 64 #number of rows and columns to convert the images to
input_shape = (img_rows, img_cols, 3)#format to store the images (rows, columns,channels) called channels last

n_of_training_images=50
for progress in range(n_of_training_images):
    # Print out progress whenever progress is a multiple of 20 so we can follow the
    # (relatively slow) progress
    if(progress%20==0):
        print(progress)
    if not split_urls[progress] == None:
        try:
            I = url_to_image(split_urls[progress])
            #check if the image has width, length and channels
            if (len(I.shape))==3:
                #create a name of each image
                save_path = os.path.join(image_base_folder,str(progress)+'.jpg')
                cv2.imwrite(save_path,I)
        except:
            None



urls_map = os.path.join(image_base_folder,'imagenet_fall11_urls','fall11_urls.txt')

txt=[]
with open(urls_map,'rb') as f:
    for line in f:
        txt.append(line)


small = txt[:500]


