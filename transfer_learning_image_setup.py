# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:49:57 2019

@author: harry
"""

import os
#import tensorflow as tf
import numpy as np
from shutil import copyfile, move
from sklearn.model_selection import train_test_split

# Input paths
img_path_dogs ='D:\Documents\PythonDoc\Photo_classification\Images\StanfordDogs'
img_path_caltech = r'D:\Documents\PythonDoc\Photo_classification\Images\Caltech256\256_ObjectCategories'
img_path_staging = r'D:\Documents\PythonDoc\Photo_classification\Images\Processed\_staging'

# Output paths
img_path_train = r'D:\Documents\PythonDoc\Photo_classification\Images\Processed\train'
img_path_valid = r'D:\Documents\PythonDoc\Photo_classification\Images\Processed\validation'
img_path_test = r'D:\Documents\PythonDoc\Photo_classification\Images\Processed\test'


# Get list of all images in a folder
def retrieve_images(folder,image_extensions=('.jpg','.jpeg', '.bmp', '.png', '.gif')):
    base_images = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    return base_images

# Extract images from Stanford Dogs and Caltech datasets and save to an intermediate folder
def send_images_to_staging(in_folder, out_folder, dataset_dog=False, n='all'):

    # Get all subfolders containing images
    subfolders = [f.path for f in os.scandir(in_folder) if f.is_dir()]

    for i, f in enumerate(subfolders):

        # Get current label of image from subfolder name
        temp = f.rsplit('\\', 1)[1]
        image_label = temp[temp.rfind('.')+1:]

        # For testing
        if n!='all':
            if i>n: break

        print("Performing subfolder " + str(i+1) + " of " + str(len(subfolders)))

        # Get images in subfolder
        f_images = retrieve_images(f)

        # Determine whether dog (image label is dog or dataset_dog set to True)
        if dataset_dog:
            is_dog = True
        else:
            is_dog = f.rsplit('.', 1)[1].lower()=='dog'

        # String to append to filename. Class label in square brackets, image
        # label in normal brackets.
        append_to_fname = '[' + ('dog' if is_dog else 'nodog') + '](' + image_label + ')'

        # Copy each image with modified filename
        for p in f_images:
            fname = p.rsplit('.')[0] + append_to_fname + '.' + p.rsplit('.')[1]
            copyfile(os.path.join(f,p), os.path.join(out_folder,fname))


send_images_to_staging(img_path_caltech, img_path_staging, dataset_dog=False, n='all')
send_images_to_staging(img_path_dogs, img_path_staging, dataset_dog=True, n='all')

# Get all images in staging folder with class label
all_staged = retrieve_images(img_path_staging)
class_label = np.array([1 if '[dog]' in x else 0 for x in all_staged])

# Check imbalance in class distribution
n_true = np.sum(class_label)
n_false = class_label.shape[0]-n_true
class_ratio = n_true/n_false

# Get a training set
X_img_train, tmp_X_valid_test, y_img_train, tmp_y_valid_test = train_test_split(all_staged,
                                                                    class_label,
                                                                    test_size=0.3,
                                                                    random_state=12)

# Get a validation and test set from the leftovers
X_img_valid, X_img_test, y_img_valid, y_img_test = train_test_split(tmp_X_valid_test,
                                                                    tmp_y_valid_test,
                                                                    test_size=0.5,
                                                                    random_state=12)

# Save as key value pairs (easier for debugging)
train_dict = dict(zip(X_img_train, y_img_train))
valid_dict = dict(zip(X_img_valid, y_img_valid))
test_dict = dict(zip(X_img_test, y_img_test))

#def clear_folder_of_images(folder):
#    print("Please confirm you want to delete all image files in '" + folder + "'.")
#    response = input("YES to delete them:\n\n")
#    if response=='YES':
#        print("Deleting image files")
#    else:
#        print("Not deleteing image files")
#
#clear_folder_of_images('this_folder')


# Move selected images from staging folder to their 'final' folder
def move_staging_to_train_test(img_dict, staging_folder, \
                               output_folder, copy=True):

    count = 0
    # For each image, save to appropriate subfolder
    for img_nm, img_lbl in img_dict.items():

        # File path and name of current image
        input_filepath = os.path.join(staging_folder, img_nm)

        # Determine folder
        if img_lbl==1:
            output_final_filepath = os.path.join(output_folder,'dog', img_nm)
        else:
            output_final_filepath = os.path.join(output_folder, 'nodog', img_nm)

        # Either copy or cut and paste image file
        if copy:
            copyfile(input_filepath, output_final_filepath)
        else:
            move(input_filepath, output_final_filepath)
        count += 1

    print("Finished moving " + str(count) + " images.")



move_staging_to_train_test(train_dict, img_path_staging, \
                               img_path_train, copy=False)

move_staging_to_train_test(valid_dict, img_path_staging, \
                               img_path_valid, copy=False)

move_staging_to_train_test(test_dict, img_path_staging, \
                               img_path_test, copy=False)



