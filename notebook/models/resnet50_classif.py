# From https://www.kaggle.com/meaninglesslives/using-resnet50-pretrained-model-in-keras/notebook

from keras.applications import ResNet50
from keras import models
from keras import layers


def get_resnet50_classif(input_shape, weights='imagenet'):

    model_base = ResNet50(include_top=False, weights=weights)
    
    Inputs = layers.Input(shape=input_shape)
    x = model_base (Inputs)
    x = layers.Flatten() (x)
    x = layers.Dense(128, activation='relu') (x)
    Outputs = layers.Dense(1, activation='sigmoid') (x)

    return models.Model(inputs=[Inputs], outputs=[Outputs])
