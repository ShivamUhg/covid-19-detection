import numpy as np 
import pandas as pd 
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16 as PTModel
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda, AvgPool2D
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
from vis.visualization import visualize_saliency
from vis.utils import utils
import numpy as np
import matplotlib.image as mpimg
import argparse
import cv2
import os
%matplotlib inline


!gdown https://drive.google.com/open?id=1odxJF4kyHEtBqhkvz3iXpV3iQK34m6z0
!unzip covid_data_compiled_sagar.zip



data_list = os.listdir('multi_class/train')
DATASET_PATH  = 'multi_class/train'
test_dir =  'multi_class/test'
IMAGE_SIZE    = (150, 150)
NUM_CLASSES   = len(data_list)
BATCH_SIZE    = 10  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 80
LEARNING_RATE =0.00001



def augment(training=True):
    

    # Train Image Augmentation
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=50,
                                       featurewise_center = True,
                                       featurewise_std_normalization = True,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.25,
                                       zoom_range=0.1,
                                       zca_whitening = True,
                                       channel_shift_range = 20,
                                       horizontal_flip = True ,
                                       vertical_flip = True ,
                                       validation_split = 0.2,
                                       fill_mode='constant')

    
    if training == True:
        

        batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                          target_size=IMAGE_SIZE,
                                                          shuffle=True,
                                                          batch_size=BATCH_SIZE,
                                                          subset = "training",
                                                          seed=42,
                                                          class_mode="categorical"
                                                          )
    else:
        
        batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                  target_size=IMAGE_SIZE,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  subset = "validation",
                                                  seed=42,
                                                  class_mode="categorical"
                                                  )
        

    return batches
