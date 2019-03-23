# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:35:20 2019

@author: harry
"""

import os
import gc
from shutil import copyfile, move

import numpy as np
import matplotlib.pyplot as plt

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
from tqdm import tqdm

# Get images in a folder based on extensions
def retrieve_images(folder,image_extensions=('.jpg','.jpeg', '.bmp', '.png', '.gif')):
    return [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]

def get_image_probabilities(img_path=None, image_list=None, batch_size=1000, action=None):

    if img_path!=None:
        # Retrieve images in folder if a path has been supplied
        apply_images_list = retrieve_images(img_path)
        path = 1
    elif image_list!=None:
        # Otherwise use the images in the list
        apply_images_list = image_list
        path = 2
    else:
        raise ValueError("Either img_path or image_list must be specified")

    # Determine number of batches which are necessary
    total_images = len(apply_images_list)
    n_batches = int(np.ceil(total_images/batch_size))

    # Initisalise empty lists (and numpy array) to hold names and probabilities
    # for each image in ALL batches. Preallocate numpy array memory rather than
    # appending.
    all_image_names = []
    all_image_probs = np.empty(total_images)
    all_images_fail = []

    # For each batch, read in images, preprocess them, put through network to
    # get probabilities.
    for nb in tqdm(range(n_batches)):

        # Indices of images in this batch
        lb = batch_size*nb
        ub = np.min([batch_size*(nb+1),total_images])

        batch_images = apply_images_list[lb:ub]

        out_batch_image = []
        out_batch_suc = []
        out_batch_fail = []

        # For each image in batch, load, process, and reshape image array
        for img_name in batch_images:
            try:
                if path==1:
                    # If a folder was supplied, add folder path to filename
                    img = image.load_img(os.path.join(img_path, img_name), target_size=(x_pixels, y_pixels))
                elif path==2:
                    # If list of images was supplied, read directly from list elements
                    img = image.load_img(img_name, target_size=(x_pixels, y_pixels))

                img = image.img_to_array(img)
                img = preprocess_input(np.expand_dims(img, axis=0))
                out_batch_suc.append(img_name)
                out_batch_image.append(img)
            except:
                # If error, append image array to list of failures and for now,
                # set image as all 0s
                temp_image = np.zeros((1, x_pixels, y_pixels, channels))
                out_batch_fail.append(img_name)
                out_batch_suc.append(img_name)
                out_batch_image.append(temp_image)

        # Stack up images list to pass for prediction
        out_batch_image = np.vstack(out_batch_image)
        batch_predictions_rev = model.predict(out_batch_image, batch_size=100)
        batch_predictions = 1-batch_predictions_rev

        # Append batch
        all_image_names += batch_images
        all_image_probs[lb:ub] = np.squeeze(batch_predictions)
        all_images_fail += out_batch_fail

        # Not hugely necessary, but potential for out_batch_image to be huge
        # so delete from memory
        del out_batch_image
        _ = gc.collect()

    return all_image_names, all_image_probs, all_images_fail


def show_rand_apply(image_names, image_probabilities, img_path=None, index=None):

    # Get random image if no index is supplied
    if index==None:
        check_image_id = np.random.randint(0,len(image_probabilities))
    elif index>len(image_probabilities):
        raise ValueError("There are only " + str(len(image_probabilities)) + "images to display")
    elif index<0:
        raise ValueError("Index must be greater than or equal to zero")
    else:
        check_image_id = index

    print("Checking image number: " +str(check_image_id) + " of " +str(len(image_probabilities)))

    # Get image and path to it
    check_image_name = image_names[check_image_id]
    if img_path!=None:
        check_image_path = os.path.join(img_path, check_image_name)
    else:
        check_image_path = check_image_name
    # Get probability and true class
    check_image_prob = image_probabilities[check_image_id]

    # Plot text as green if correct prediction (using threshold of 0.5), or red
    # if incorrect
    check_image_color = '#40e843' if check_image_prob>=0.5 else '#ea1212'

    # Load image and define text to be displayed with it
    try:
        check_image = image.load_img(check_image_path)
        check_text = "Prob. est.: {0:.1%}".format(check_image_prob)

        # Set up for showing image
        box_style = {'boxstyle': 'round',
                     'facecolor':'black',
                     'alpha':0.7}
        x_text = 0.55
        y_text = 0.95

        # Show image
        fig, ax = plt.subplots(figsize=(7,7))
        plt.imshow(check_image)
        plt.axis('off')
        plt.text(x_text, y_text, check_text, size=17, color=check_image_color, fontweight='bold',
                 transform=ax.transAxes, verticalalignment='top', bbox=box_style)
        plt.tight_layout()
    except:
        print("Cannot open image: " + check_image_name)


# Select images with probability over a certain threshold
def select_images(img_names, img_probs, threshold=0.5):
    return [img for i, img in enumerate(img_names) if img_probs[i]>=threshold]

# Text file containing image names and their probabilities
def save_image_text(img_names, img_probs, outfile):
    with open(outfile,'w') as f:
        for i, img in enumerate(img_names):
            f.write(img + ',' + str(img_probs[i]) + '\n')

# Copy/cut images in a list from one location to another
def copy_images(img_names, output_path, img_path=None, action='copy',
                             save_with_prob=False):

    for img in img_names:

        # Set input filename, either with a path in front if specified, or not
        if img_path!=None: infile = os.path.join(img_path,img)
        else: infile = img

        # Get image name without any path
        img_no_path = infile.rsplit('\\', 1)[1]

        # Set output filename with output path
        outfile = os.path.join(output_path, img_no_path)

        if action=='copy':
            copyfile(infile, outfile)
        elif action=='cut':
            move(infile, outfile)
        else:
            raise ValueError("action must be 'copy' or 'cut'")

# Text file containing image names and their probabilities
def save_image_text2(img_names, img_probs, outfile):
    with open(outfile,'w') as f:
        for i, img in enumerate(img_names):
            img_no_path = img.rsplit('\\', 1)[1]

            f.write(img_no_path + ',' + str(img_probs[i]) + '\n')


def rec_folder_search(base_folder):
    all_images = []
    for root, subdirs, files in os.walk(base_folder):

        for folder in subdirs:
            folder_images = [os.path.join(root, folder, x) for x in retrieve_images(os.path.join(root, folder))]
            all_images += folder_images

    return all_images

# Folder where final modelled as true images are stored
output_path=r'D:\Documents\PythonDoc\Photo_classification\Images\test_final_output'

# Define size of image (channels=greyscale (1) or colour(3))
x_pixels = 200
y_pixels = 200
channels = 3

batch_size = 100

# choose a model name to load and its location
model_name = 'dropout_tl.h5'
model_location = 'D:\Documents\PythonDoc\Photo_classification\models'

# Threshold for classifying as yes
prob_threshold = 0.5

# Load in network
model = load_model(os.path.join(model_location, model_name))

# =============================================================================
# Images in one directory - no subdirectories
# =============================================================================
# Folder where images are stored
img_path = r'D:\Documents\PythonDoc\Photo_classification\Images\Test_images'

# Get probabilities for each image
img_names, img_probs, img_fails = get_image_probabilities(img_path=img_path, batch_size=batch_size, action=None)

# Sample of images and probabilities
for i in range(10):
    show_rand_apply(img_names, img_probs, img_path=img_path)

# Probability distribution
plt.figure()
plt.hist(img_probs, bins=20, density=True)
plt.show()

# Get those images which are true accoreding to model
selected_images = select_images(img_names, img_probs, threshold=0.5)

# Save a text file of all image names and their probabilities
save_image_text(img_names, img_probs, outfile=r'D:\Documents\PythonDoc\Photo_classification\Images\image_probs.txt')

# Copy 'true' pictures into another location
copy_images(img_path=img_path, selected_images, output_path=output_path, action='copy',
                             save_with_prob=False)


# =============================================================================
# Images in directory - with subdirectories
# =============================================================================
batch_size = 1000
root_folder = r'F:\Photo_backup_20181130\From_MAC'
final_folder = r'F:\dog_images'

# Get all image files in all subdirectories with full path
images_all_folders = rec_folder_search(root_folder)

# Get probabilities for each image
img_names, img_probs, img_fails = get_image_probabilities(image_list=images_all_folders,
                                                            batch_size=batch_size, action=None)

out_imgs = []
for i, img in enumerate(img_names):
    if img_probs[i]>=0.9:
        out_imgs += [img]



save_image_text2(img_names, img_probs, outfile=r'D:\Documents\PythonDoc\Photo_classification\Images\image_probs.txt')

# Get those images which are true accoreding to model
selected_images = select_images(img_names, img_probs, threshold=0.5)

# Copy 'true' pictures into another location
copy_images(out_imgs, final_folder, action='copy',
                             save_with_prob=False)


show_rand_apply(img_names, img_probs, img_path=None, index=7807)

for i in range(10):
    show_rand_apply(img_names, img_probs, img_path=None)


