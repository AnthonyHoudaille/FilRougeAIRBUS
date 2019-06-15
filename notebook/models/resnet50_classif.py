# From https://www.kaggle.com/meaninglesslives/using-resnet50-pretrained-model-in-keras/notebook
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

import keras 
from keras.applications import ResNet50
from keras import models
from keras import layers
from keras import backend as K


def get_resnet50_classif(input_shape, weights='imagenet'):
    K.clear_session()
    model_base = ResNet50(include_top=False, weights=weights)
    
    Inputs = layers.Input(shape=input_shape, name='Input_Image')
    x = model_base (Inputs)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    Outputs = layers.Dense(1, activation='sigmoid', name='fc1') (x)

    return models.Model(inputs=[Inputs], outputs=[Outputs])
