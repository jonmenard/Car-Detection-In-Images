# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:49:53 2022

@author: Jon Menard
"""

# %% Import all required libraries
import cv2
import shutil
import os
import pandas as pd
import numpy as np
from numpy import expand_dims,asarray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from IPython.display import SVG
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Conv2D,MaxPool2D,Activation,Concatenate,MaxPooling2D,Input, GlobalAveragePooling2D,GlobalMaxPooling2D, AveragePooling2D   
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.layers.merge import concatenate
from PIL import Image
gpus = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpus, True)
img_width, img_height = 225, 400 


# %% Code for implementing the Custom CNN

inputLayer = Input(shape=(img_width,img_height,3))

x = Conv2D(32, (3,4), padding="same", activation="relu")(inputLayer)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = AveragePooling2D(pool_size=(2,3))(x) 

x = Conv2D(32, (3,3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = MaxPooling2D(pool_size=(3,3))(x) 

x = Conv2D(64, (3,3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = MaxPooling2D(pool_size=(3,3))(x)

x = Conv2D(64, (2,2), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(128, (2,2), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = MaxPooling2D(strides=(2,2))(x)

x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(256,activation="relu")(x)
outputLayer = Dense(1,activation="sigmoid")(x)

model = keras.Model(inputLayer, outputLayer)


# Adam with inital learning rate for training 0.0002
model.compile(optimizer= keras.optimizers.Adam(learning_rate= 0.0002), loss= 'binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)
model.save('./.trainedModel.hdf5') # save the model
# %% Save the models weights
model.save_weights('weights.h5')
# %% recompile the model with the lower learning rate for further training
model.compile(optimizer= keras.optimizers.Adam(learning_rate= 0.0000025), loss= 'binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy()])
model.load_weights('weights.h5') # relode the semi trained models weightsd back into the model

# %% Loading the datasets
# This is the directory containing the training data

train_data_dir = "../Dataset/Dataset/trainingData/"
img_height, img_width = 225, 400 
channels = 3
batch_size = 32


#Getting Dataset
train_datagen = ImageDataGenerator(
    rescale= 1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip= True,
    shear_range=0.2,
    zoom_range=0.2,
    validation_split=0.2,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(  
    train_data_dir,  
    target_size= (img_height, img_width), 
    color_mode= 'rgb',
    batch_size= batch_size,  
    class_mode= 'binary',
    subset='training',
    shuffle= True, 
    seed= None
)

valid_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size= (img_height, img_width), 
    color_mode= 'rgb',
    batch_size= batch_size,  
    class_mode= 'binary',
    subset='validation',
    shuffle= True, 
    seed= 1337
) 

num_classes = len(train_generator.class_indices)  
train_labels = train_generator.classes 
train_labels = to_categorical(train_labels, num_classes=num_classes)

valid_labels = valid_generator.classes 
valid_labels = to_categorical(valid_labels, num_classes=num_classes)
nb_train_samples = len(train_generator.filenames)  
nb_valid_samples = len(valid_generator.filenames)


# %% Training the model
# Adding a checkpoint to save the model everytime a new best validation accuracy is obtained
mcp_save = ModelCheckpoint('.modelCheckpoint.hdf5', save_best_only=True, monitor='val_loss', mode='auto')
          
with tf.device('/gpu:0'):
    history = model.fit(
        train_generator, 
        epochs = 75,
        batch_size = batch_size,
        steps_per_epoch = nb_train_samples//batch_size,
        validation_data = valid_generator, 
        validation_steps = nb_valid_samples//batch_size,
        verbose = 2,
        initial_epoch = 0,
        callbacks = mcp_save,
        shuffle = True
    )
  
if(False): #Change to true for final validation accuracy    
    (eval_loss, eval_accuracy) = model.evaluate(valid_generator, batch_size= batch_size, verbose= 1)
    print('The final Validation Loss is: ', eval_loss)
    print('The final Validation Accuracy is: ', eval_accuracy)

# %% Plotting Training History

print("The best val_accuarcy was" , max(history.history['val_binary_accuracy']))

plt.subplot()
plt.title('Model Accuracy')
plt.plot(history.history['binary_accuracy'][0:100])
plt.plot(history.history['val_binary_accuracy'][0:100])
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training Accuracy','Validation Accuracy'])
plt.savefig('baseline_acc_epoch.png', transparent= False, bbox_inches= 'tight', dpi= 900)
plt.show()

plt.title('Model Loss')
plt.plot(history.history['loss'][0:100])
plt.plot(history.history['val_loss'][0:100])
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training Loss','Validation Loss'])
plt.savefig('baseline_loss_epoch.png', transparent= False, bbox_inches= 'tight', dpi= 900)
plt.show()