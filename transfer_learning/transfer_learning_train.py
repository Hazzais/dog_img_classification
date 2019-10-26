# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:04:53 2019

@author: harry
"""
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model


def print_layers(inmodel):
    for i_layer, v_layer in enumerate(inmodel.layers):
        print(i_layer, v_layer.name)


if __name__ == '__main__':

    # TODO: args?
    # TODO: model type argument (e.g. ResNet50 etc.)
    # TODO: add dropout layers?
    # TODO: add validation data into fit model
    # Define size of image (channels=greyscale (1) or colour(3))
    x_pixels = 200
    y_pixels = 200
    channels = 3

    img_flow_batch_size = 32
    train_epochs = 10

    # Base path of images - assumes this contains train, test, val folder
    # with positive and negative class folders within each (named identically)
    img_path = r'D:\Documents\PythonDoc\Photo_classification\Images\Processed'

    # choose a name to save the model as
    model_out_path = os.path.join('models', 'dropout_tl.h5')

    # =========================================================================
    # Define model
    # =========================================================================
    # Load the ResNet50 model for transfer learning without output layer
    model = ResNet50(weights='imagenet',
                     input_shape=(x_pixels, y_pixels, channels),
                     include_top=False
                     )

    # Define architecture. Used the following as a starting point:
    # https://towardsdatascience.com/keras-transfer-learning-for-beginners-6c9b8b7143e
    arch = model.output
    arch = GlobalAveragePooling2D()(arch)
    arch = Dense(1024, activation='relu')(arch)
    arch = Dropout(0.2)(arch)
    arch = Dense(1024, activation='relu')(arch)
    arch = Dropout(0.2)(arch)
    arch = Dense(512, activation='relu')(arch)
    arch = Dropout(0.2)(arch)
    pred = Dense(1, activation='sigmoid')(arch)

    # Combine the ResNet model inputs and the architecture defined above
    model_use = Model(inputs=model.input, outputs=pred)

    # Make ResNet model layers not trainable so we may use their learned
    # features
    for layer in model.layers:
        layer.trainable = False

    # =========================================================================
    # Load images
    # =========================================================================

    # Training data
    datagen_train =\
        ImageDataGenerator(preprocessing_function=preprocess_input)
    data_train =\
        datagen_train.flow_from_directory(os.path.join(img_path, 'train'),
                                          target_size=(x_pixels, y_pixels),
                                          batch_size=img_flow_batch_size,
                                          class_mode='binary',
                                          color_mode='rgb',
                                          shuffle=True)

    # TODO: Check this - move parts to eval file?
    datagen_test =\
        ImageDataGenerator(preprocessing_function=preprocess_input)
    # Validation data - for use when fitting to evaluate on the go
    data_test_infit =\
        datagen_test.flow_from_directory(os.path.join(img_path, 'test'),
                                         target_size=(x_pixels, y_pixels),
                                         batch_size=img_flow_batch_size,
                                         class_mode='binary',
                                         color_mode='rgb',
                                         shuffle=True)

    # Validation data - for use after model fitted. Use batch size of one and
    # no shuffle as this allows easy mapping back to filenames.
    data_test =\
        datagen_test.flow_from_directory(os.path.join(img_path, 'test'),
                                         target_size=(x_pixels, y_pixels),
                                         batch_size=1,
                                         class_mode='binary',
                                         color_mode='rgb',
                                         shuffle=False)

    class_label_order = data_train.class_indices

    # =========================================================================
    # Compile and train network
    # =========================================================================
    # Binary classification
    model_use.compile(optimizer='Adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    # Number of steps needed per epoch is the number needed for each image to
    # go through, batch_size at a time
    step_size_train = data_train.n//data_train.batch_size

    # Train the model - the long bit. Fit on training data generator and
    # provide validation on the validation data set.
    model_history = model_use.fit_generator(generator=data_train,
                                            steps_per_epoch=step_size_train,
                                            epochs=train_epochs,
                                            validation_data=data_test_infit)

    # Save model for later so it does not have to be retrained
    model_use.save(model_out_path)
