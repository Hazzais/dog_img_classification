# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 19:16:35 2019

@author: harry
"""
import os
import argparse
import warnings

from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import urllib


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download a number of images "
                                                 "for a WordNet Id into a "
                                                 "folder")
    parser.add_argument('wordnet_id', type=str,
                        help='The WordNet Id to download images for')
    parser.add_argument('out_path', type=str,
                        help='The location to save images')
    parser.add_argument('-n', '--n-images', default=50, type=int,
                        help='The number of images to attempt to download')
    parser.add_argument('-x', '--x-pixels', default=64, type=int,
                        help="Number of x-pixels to convert each image to")
    parser.add_argument('-y', '--y-pixels', default=64, type=int,
                        help="Number of y-pixels to convert each image to")
    parser.add_argument('-f', '--file-extension', default='jpg',
                        choices=['jpg', 'png'],
                        help="Extension to save images with")
    parser.add_argument('-b', '--base_url', type=str,
                        default='http://www.image-net.org/api/text/'
                                'imagenet.synset.geturls?wnid=',
                        help='The number of x pixels to convert the image to')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print progress every 20 images")
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    img_url = "{}{}".format(args.base_url, args.wordnet_id)

    page = requests.get(img_url)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Need to be able to strip the soup so individual URLs can be different
    # elements of a list
    str_soup = str(soup)
    split_urls = str_soup.split('\r\n')

    image_base_folder = args.out_path
    img_extension = args.file_extension

    img_rows, img_cols = args.x_pixels, args.y_pixels

    # Third argument is number of channels - keep as 3 to keep colour
    input_shape = (img_rows, img_cols, 3)

    n_of_training_images = args.n_images

    imgs_no_load = []
    for i_img in range(n_of_training_images):

        # Optionally print out i_img whenever i_img is a multiple of 20
        # so we can follow the (relatively slow) process
        if i_img % 20 == 0:
            print(i_img)
        if split_urls[i_img]:
            try:
                img_contents = url_to_image(split_urls[i_img])
                # check if the image has width, length and channels
                if len(img_contents.shape) == 3:
                    # create a name of each image
                    save_path = os.path.join(image_base_folder,
                                             '{}.{}'.format(i_img,
                                                            img_extension))
                    cv2.imwrite(save_path, img_contents)
            except:
                imgs_no_load.append(split_urls[i_img])

    no_load = len(imgs_no_load)
    if no_load > 0:
        warnings.warn("There were {} images which could not be downloaded"
                      .format(no_load))
