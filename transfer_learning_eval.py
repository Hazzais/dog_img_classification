import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


model_name = os.path.join('models', 'dropout_tl.h5')
img_path = r'D:\Documents\PythonDoc\Photo_classification\Images\Processed'
x_pixels = 200
y_pixels = 200

# =============================================================================
# Evaluation
# =============================================================================
# TODO: turn into a Jupyter notebook?

model_use = load_model(model_name)

# Get predictions and show some results.
# https://stackoverflow.com/questions/45806669/keras-how-to-use-predict-generator-with-imagedatagenerator

datagen_test = \
    ImageDataGenerator(preprocessing_function=preprocess_input)
data_test = \
    datagen_test.flow_from_directory(os.path.join(img_path, 'test'),
                                     target_size=(x_pixels, y_pixels),
                                     batch_size=1,
                                     class_mode='binary',
                                     color_mode='rgb',
                                     shuffle=False)

filenames = data_test.filenames
true_class = data_test.classes
true_class_rev = 1 - true_class
total_test_images = len(filenames)
pred_prob = model_use.predict_generator(data_test, steps=total_test_images,
                                        verbose=1)
pred_prob_rev = 1 - pred_prob

# Prediction probability for each image being a dog
results_dict = dict(zip(filenames, pred_prob_rev))

# Confusion matrix
conf_mat = confusion_matrix(true_class, pred_prob > 0.5)
print(conf_mat)

# Get ROC AUC score and arrays for building ROC curve
fpr, tpr, thresholds = roc_curve(true_class, pred_prob)
auc = roc_auc_score(true_class, pred_prob > 0.5)

# ROC curve
plt.figure()
plt.plot(np.linspace(0, 1, num=50), '--', color='gray')
plt.plot(fpr, tpr, '-', color='red')
plt.xlabel('FPR', fontsize=14)
plt.ylabel('TPR', fontsize=14)
plt.show()


# Get a random image in the test data and display it with it's associated
# predicted probability.
def rand_check(index=None):
    # Get random image if no index is supplied
    if index == None:
        check_image_id = np.random.randint(0, len(pred_prob_rev))
    elif index > len(pred_prob_rev):
        raise ValueError(
            "There are only " + str(len(pred_prob_rev)) + "images to display")
    elif index < 0:
        raise ValueError("Index must be greater than or equal to zero")
    else:
        check_image_id = index

    # Get image and path to it
    check_image_name = filenames[check_image_id]
    check_image_path = os.path.join(img_path, 'test', check_image_name)

    # Get probability and true class
    check_image_prob = results_dict[check_image_name]
    check_image_true = true_class_rev[check_image_id]

    # Plot text as green if correct prediction (using threshold of 0.5), or red
    # if incorrect
    check_image_color = '#40e843' if \
        ((check_image_prob >= 0.5) and (check_image_true == 1)) or (
                    (check_image_prob < 0.5) and (
                        check_image_true == 0)) else '#ea1212'

    # Load image and define text to be displayed with it
    check_image = image.load_img(check_image_path)
    check_text = "Prob. est.: {0:.1%}".format(check_image_prob[0])

    # Set up for showing image
    box_style = {'boxstyle': 'round',
                 'facecolor': 'black',
                 'alpha': 0.7}
    x_text = 0.55
    y_text = 0.95

    # Show image
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.imshow(check_image)
    plt.axis('off')
    plt.text(x_text, y_text, check_text, size=17, color=check_image_color,
             fontweight='bold',
             transform=ax.transAxes, verticalalignment='top', bbox=box_style)
    plt.tight_layout()


rand_check()

# Get indices of true positives:
tp_mask = ((true_class_rev == 1) & np.squeeze(pred_prob_rev >= 0.5))
tp_indices = np.squeeze(np.transpose(np.nonzero(tp_mask)))

# Get indices of true negatives:
tn_mask = ((true_class_rev == 0) & np.squeeze(pred_prob_rev < 0.5))
tn_indices = np.squeeze(np.transpose(np.nonzero(tn_mask)))

# Get indices of false positives:
fp_mask = ((true_class_rev == 0) & np.squeeze(pred_prob_rev >= 0.5))
fp_indices = np.squeeze(np.transpose(np.nonzero(fp_mask)))

# Get indices of false negatives:
fn_mask = ((true_class_rev == 1) & np.squeeze(pred_prob_rev < 0.5))
fn_indices = np.squeeze(np.transpose(np.nonzero(fn_mask)))

# Random true positive
rand_check(np.random.choice(tp_indices))

# Random true negative
rand_check(np.random.choice(tn_indices))

# Random false positive
rand_check(np.random.choice(fp_indices))

# Random false negative
rand_check(np.random.choice(fn_indices))

rand_choice_tp = np.random.choice(tp_indices, size=10, replace=False)
rand_choice_tn = np.random.choice(tn_indices, size=10, replace=False)
rand_choice_fp = np.random.choice(fp_indices, size=10, replace=False)
rand_choice_fn = np.random.choice(fn_indices, size=10, replace=False)


def show_results(tp_ind, tn_ind, fp_ind, fn_ind, n_tp=5, n_tn=5, n_fp=5,
                 n_fn=5, n_cols=5):
    n_imgs = n_tp + n_tn + n_fp + n_fn
    n_rows = int(np.ceil((n_imgs) / n_cols))

    def add_type(inds, ns, string):
        for x in range(min(len(inds), ns)):
            x_this_ind = inds[x]

            img_name = filenames[x_this_ind]
            all_imgs.append(
                image.load_img(os.path.join(img_path, 'test', img_name)))
            all_probs.append(pred_prob_rev[x_this_ind])
            all_title.append(
                string + ": {0:.1%}".format(pred_prob_rev[x_this_ind][0]))
            all_type.append(string)

    all_imgs = []
    all_probs = []
    all_type = []
    all_title = []

    add_type(tp_ind, n_tp, 'TP')
    add_type(tn_ind, n_tn, 'TN')
    add_type(fp_ind, n_fp, 'FP')
    add_type(fn_ind, n_fn, 'FN')

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(10, 8), sharex=True,
                           sharey=True)
    img_count = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if img_count < n_imgs:
                #                ax[r,c].imshow(all_imgs[img_count], interpolation='nearest', aspect='auto')
                ax[r, c].imshow(all_imgs[img_count], interpolation="nearest")
                ax[r, c].autoscale(False)
                ax[r, c].yaxis.set_ticklabels([])
                ax[r, c].yaxis.set_ticks_position('none')
                ax[r, c].xaxis.set_ticklabels([])
                ax[r, c].xaxis.set_ticks_position('none')
                border_color = '#40e843' if (all_type[img_count] == 'TP') or (
                            all_type[img_count] == 'TN') else '#ea1212'
                # https://stackoverflow.com/questions/7778954/elegantly-changing-the-color-of-a-plot-frame-in-matplotlib
                for spine in ax[r, c].spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                ax[r, c].set_title(all_title[img_count])
                img_count += 1


show_results(rand_choice_tp, rand_choice_tn, rand_choice_fp, rand_choice_fn,
             n_tp=3, n_tn=3, n_fp=3, n_fn=3, n_cols=3)

# =============================================================================
# Apply
# =============================================================================
# TODO: make into another apply script?
img_path_apply = 'D:\Documents\PythonDoc\Photo_classification\Images\Holiday 2015'


# Get list of all images in a folder
def retrieve_images(folder, image_extensions=(
'.jpg', '.jpeg', '.bmp', '.png', '.gif')):
    base_images = [f for f in os.listdir(folder) if
                   f.lower().endswith(image_extensions)]
    return base_images


apply_images_list = retrieve_images(img_path_apply)

# https://gist.github.com/ritiek/5fa903f97eb6487794077cf3a10f4d3e
images = []
images_name = []
for tmp, img in enumerate(apply_images_list):
    images_name.append(img)
    img = image.load_img(os.path.join(img_path_apply, img),
                         target_size=(x_pixels, y_pixels))
    img = image.img_to_array(img)
    img = preprocess_input(np.expand_dims(img, axis=0))
    images.append(img)

# stack up images list to pass for prediction
images = np.vstack(images)
predictions_apply = model_use.predict(images, batch_size=10)
pred_prob_rev_apply = 1 - predictions_apply


def show_rand_apply(index=None):
    # Get random image if no index is supplied
    if index == None:
        check_image_id = np.random.randint(0, len(pred_prob_rev_apply))
    elif index > len(pred_prob_rev_apply):
        raise ValueError("There are only " + str(
            len(pred_prob_rev_apply)) + "images to display")
    elif index < 0:
        raise ValueError("Index must be greater than or equal to zero")
    else:
        check_image_id = index

    # Get image and path to it
    check_image_name = images_name[check_image_id]
    check_image_path = os.path.join(img_path_apply, check_image_name)

    # Get probability and true class
    check_image_prob = pred_prob_rev_apply[check_image_id]

    # Plot text as green if correct prediction (using threshold of 0.5), or red
    # if incorrect
    check_image_color = '#40e843' if check_image_prob >= 0.5 else '#ea1212'

    # Load image and define text to be displayed with it
    check_image = image.load_img(check_image_path)
    check_text = "Prob. est.: {0:.1%}".format(check_image_prob[0])

    # Set up for showing image
    box_style = {'boxstyle': 'round',
                 'facecolor': 'black',
                 'alpha': 0.7}
    x_text = 0.55
    y_text = 0.95

    # Show image
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.imshow(check_image)
    plt.axis('off')
    plt.text(x_text, y_text, check_text, size=17, color=check_image_color,
             fontweight='bold',
             transform=ax.transAxes, verticalalignment='top', bbox=box_style)
    plt.tight_layout()


for i in range(20):
    show_rand_apply()
