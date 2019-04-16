# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:00:51 2019

@author: antho
"""


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from preprocess.pre_process import multi_rle_encode, rle_encode, rle_decode, masks_as_image, masks_as_color, balancing_train
from preprocess.pre_process import make_image_gen, create_aug_gen
from sklearn.model_selection import train_test_split
from keras import backend as K

import tensorflow as tf
with tf.Session() as sess:
    devices = sess.list_devices()
for device in devices:
    print(device)
    
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
    
    
ship_dir = '/mnt/data/'
train_image_dir = os.path.join(ship_dir, 'train')# Images for training
test_image_dir = os.path.join(ship_dir, 'test')# Images for testing
label_dir = os.path.join(ship_dir, 'train_ship_segmentations_v2.csv')# Images for testing
masks = pd.read_csv(label_dir, engine="python") # Markers for ships

label_has_ships = masks
label_has_ships['has_ship'] = masks.EncodedPixels.notnull()
label_has_ships = label_has_ships.drop(columns=['EncodedPixels'])
label_has_ships = label_has_ships.drop_duplicates()

count_has_ship = label_has_ships.has_ship.sum()
count_has_ship
print("rate:", count_has_ship/len(label_has_ships))


data_balanced = label_has_ships.groupby('has_ship')
data_balanced = data_balanced.apply(lambda x: x.sample(data_balanced.size().min()).reset_index(drop=True))

count_has_ship = data_balanced.has_ship.sum()
count_has_ship
print("rate:", count_has_ship/len(data_balanced))