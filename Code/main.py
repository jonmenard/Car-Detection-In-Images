# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:49:53 2022

@author: Jon
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
from skimage.transform import resize
from IPython.display import SVG
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Conv2D,MaxPool2D,Activation,Concatenate,MaxPooling2D,Input, GlobalAveragePooling2D  
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
#from tensorflow.keras.applications.VGG19 import VGG19
from tensorflow.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.layers.merge import concatenate
from PIL import Image
# Set GPU to allow memory growth
gpus = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpus, True)

# This is the directory containing the training data
train_data_dir = "../Dataset/trainingData/"
img_width, img_height = 225, 400 
channels = 3
batch_size = 16


#Getting Dataset 
train_datagen = ImageDataGenerator(
    rescale= 1./255,
    horizontal_flip= True,
)

valid_datagen = ImageDataGenerator(
    #rescale= 1./255, 
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(  
    train_data_dir,  
    target_size= (img_width, img_height), 
    color_mode= 'rgb',
    batch_size= batch_size,  
    class_mode= 'binary',
    subset='training',
    shuffle= True, 
    seed= 1337
)

valid_generator = valid_datagen.flow_from_directory(
    train_data_dir,
    target_size= (img_width, img_height),
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


# %% Code for implementing the Flat NN
inputLayer = Input(shape=(img_width,img_height,3))
x = Flatten()(inputLayer)
x = Dense(512,activation="relu")(x)
x = Dense(1024,activation="relu")(x)
x = Dense(512,activation="relu")(x)
x = Dense(256,activation="relu")(x)
x = Dense(64,activation="relu")(x)
x = Dense(16,activation="relu")(x)
outputLayer = Dense(1,activation="sigmoid")(x)
model = keras.Model(inputLayer, outputLayer)
model.compile(optimizer= keras.optimizers.Adam(learning_rate= 0.00005), loss= 'binary_crossentropy', metrics= ['accuracy'])
# %% Code for implementing the InceptionV3 CNN

InceptionV3 = InceptionV3(include_top= False, input_shape= (img_width, img_height, channels), weights= 'imagenet')
model = Sequential()
model.add(InceptionV3)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer= keras.optimizers.Adam(learning_rate= 0.00005), loss= 'binary_crossentropy', metrics= ['accuracy'])
# %% Code for implementing the Custom CNN

inputLayer = Input(shape=(img_width,img_height,3))
x = Conv2D(64, 3,strides=(1,1), padding="same", activation="relu")(inputLayer)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(strides=(2,2))(x)

x = Conv2D(256, 3, strides=(1,1), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(strides=(2,2))(x)

x = Conv2D(512, 3,strides=(1,1), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(strides=(2,2))(x)

x = Conv2D(1024, 3,strides=(1,1), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = GlobalAveragePooling2D(keepdims="True")(x)

x = Flatten()(x)
x = Dense(512,activation="relu")(x)
x = Dense(256,activation="relu")(x)
x = Dense(32,activation="relu")(x)
outputLayer = Dense(1,activation="sigmoid")(x)

model = keras.Model(inputLayer, outputLayer)

# Get the output of the first convolution layer - to show feautures
modelVis = keras.Model(inputLayer, model.layers[2].output)

model.compile(optimizer= keras.optimizers.Adam(learning_rate= 0.00005), loss= 'binary_crossentropy', metrics= ['accuracy'])
SVG(model_to_dot(model).create(prog='dot', format='svg'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, expand_nested=True)
# %% Traning

# Adding a checkpoint to save the model everytime a new best validation accuracy is obtained
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='auto')
# Traing the model with GPU, 100 epoches
with tf.device('/gpu:0'):
    history = model.fit(
        train_generator, 
        epochs = 100,
        batch_size = batch_size,
        steps_per_epoch = nb_train_samples//batch_size,
        validation_data = valid_generator, 
        validation_steps = nb_valid_samples//batch_size,
        verbose = 2, 
        callbacks = mcp_save,
        shuffle = True
    )
    
(eval_loss, eval_accuracy) = model.evaluate(valid_generator, batch_size= batch_size, verbose= 1)
print('The final Validation Loss is: ', eval_loss)
print('The final Validation Accuracy is: ', eval_accuracy)

# %% Plotting Training History

print("The best val_accuarcy was" , max(history.history['val_accuracy']))

plt.subplot()
plt.title('Model Accuracy')
plt.plot(history.history['accuracy'][0:100])
plt.plot(history.history['val_accuracy'][0:100])
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

# %% Load a saved model for the custom CNN%
model = tf.keras.models.load_model('./Custom CNN/.mdl_wts.hdf5')
# Set the model to not be trainable
for layer in model.layers:
    layer.trainable= False
modelVis = keras.Model(model.layers[0].output, model.layers[1].output)
    
# %% Load all data to be tested on - no validation split

test_data_dir = "../Dataset/test"
img_width, img_height = 225, 400 
channels = 3
batch_size = 32
image_arr_size= img_width * img_height * channels

test_datagen = ImageDataGenerator(
    rescale= 1./255,
    validation_split=0.99,
)

testing_generator = test_datagen.flow_from_directory(  
    test_data_dir,
    target_size= (img_width, img_height),
    color_mode= 'rgb',
    batch_size= 1,  
    class_mode= 'binary',
    subset='validation',
    shuffle= True, 
    seed= 1337
)
# %% Evaluation of model on all data
score = model.evaluate(testing_generator, verbose=1, batch_size=batch_size)
print("%s: %.3f%%" % (model.metrics_names[1], score[1] * 100))


# %% Model Prediction on Test Data

src = "../Dataset/test/cars/"
for filename in os.listdir(src):
    if(filename == "trainingData"):
        continue
  
for filename in os.listdir(src):
    
    image = np.asarray(load_img(src + filename, target_size=(225, 400))).astype('float32') / 255
    image = expand_dims(image, axis=0)   
    #x = model(image)
    score, acc = model.evaluate(image,np.array([0]),verbose=0, batch_size=1)
    label = "This does not have car"
    if(acc >= 1):
        label = "This has a car"
        
    plt.imshow(image[0])
    plt.title(label)
    plt.show()          

