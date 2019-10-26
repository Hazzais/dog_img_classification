# Dog Image Classification

The goal of this project is to be able to classify (generally my) photos 
depending on whether they contain dogs or not as I tend to take lots of 
pictures of my dogs!

The goal is not to create the perfect neural network architecture; I certainly 
make no claims I am an expert in deep learning! I simply want to get something 
which works at this point. However, I have a few approaches:
1) Use of a model ([ResNet50](https://www.kaggle.com/keras/resnet50)) 
pre-trained on the [ImageNet](http://www.image-net.org/) data set followed by
application of [WordNet hyponyms](https://wordnet.princeton.edu/) 
to predict whether my images contain dogs
2) Using transfer learning to tailor the pre-trained model to my specific needs
3) Training a CNN from scratch

#### Data used:
[Stanford dogs data](http://vision.stanford.edu/aditya86/ImageNetDogs/) - used 
for most of the dog images

[Caltech 256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) - used
for non-dog images (though there are some dogs as well)  

## Approaches
### 1) Pre-trained models and WordNet

* **keras_pretrained.py:** Simple script to, using the ResNet50 model, predict 
what a supplied image contains. Not specific to dogs.
* **keras_pretrained_apply.py:** Script to take a word (in my case dogs), 
use WordNet to get 'sub-words' (i.e. hyponyms or words like 'German Shepherd' 
for dogs), and calculate the probability of one or more images containing 
the word (and subsequent sub-words).

### 2) Transfer learning

* **transfer_learning_image_setup.py:** Takes images from the Stanford dogs 
dataset and the Caltech256 dataset to create image subsets for training and 
testing
* **transfer_learning_train.py:** A simple application of transfer learning to 
take the ResNet50 model and train several additional layers to tailor the 
model to predict dogs specifically. Due to the time it takes to train on my 
laptop (even using a GPU), I have not done much experimentation with the 
architecture. Maybe one day I'll try a hefty cloud instance and experiment 
further. In the meantime the ResNet50 model gets me a long way.
* **transfer_learning_eval.py:** Evaluating the model on the hold-out dataset.
* **transfer_learning_apply.py:** Applying the model to a different set of 
images!

### 3) Creating a neural network from scratch
Not currently started...

### Other files
* **additional_imagenet_batch_download.py:** File which can download additional
 images from ImageNet for a given WordNet word id
 
## Notes
The model I created is too large to upload to GitHub hence its absence. I am 
looking into storing this in another location such as a public S3 bucket. 
Similarly, storing the images in a location agnostic to the environment.

I also started a lot of this code a while ago and have only come back recently 
to clear things up. I've noticed I should probably do a fair bit of refactoring
 including moving certain shared functions to single places and improving 
several of the functions. However, quite a lot of this is me experimenting and 
performing a one-off batch of image classifications so is not meant to be 
production-grade code!