# From https://www.kaggle.com/meaninglesslives/using-resnet50-pretrained-model-in-keras/notebook
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

from tensorflow.python import keras as keras
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K


def get_resnet50_classif(input_shape, weights='imagenet'):
    K.clear_session()
    model_base = ResNet50(include_top=False, weights=weights)
    
    Inputs = layers.Input(shape=input_shape)
    x = model_base (Inputs)
    x = layers.Flatten() (x)
    x = layers.Dense(128, activation='relu') (x)
    Outputs = layers.Dense(1, activation='sigmoid') (x)

    return models.Model(inputs=[Inputs], outputs=[Outputs])



def get_resnet50_classif_v2(input_shape, weights='imagenet', input_tensor=None):
    K.clear_session()
   
    if tensor is None:
        model_base = ResNet50(include_top=False, weights=weights, input_tensor=None, input_shape=input_shape)
    else:
        model_base = ResNet50(include_top=False, weights=weights, input_tensor=input_tensor, input_shape=input_shape)
    x = layers.Flatten() (model_base)
    x = layers.Dense(128, activation='relu') (x)
    Outputs = layers.Dense(1, activation='sigmoid') (x)

    return models.Model(inputs=[model_base], outputs=[Outputs])