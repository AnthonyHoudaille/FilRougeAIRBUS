# From https://www.kaggle.com/meaninglesslives/using-resnet50-pretrained-model-in-keras/notebook

from keras.applications import ResNet50
from keras import models
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout, Activation 
from keras.layers import BatchNormalization, UpSampling2D, Dropout, AvgPool2D, SpatialDropout2D

def decoder_block(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (1, 1), padding="same", strides=strides, name=prefix + "_conv1")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn1")(conv)
    conv = Conv2DTranspose(filters, (4, 4), strides=(2, 2), padding='same', name=prefix + "_Transp_conv")
    conv = BatchNormalization(name=prefix + "_bn2")(conv)
    conv = Conv2D(filters, (1, 1), padding="same", strides=strides, name=prefix + "_conv2")(conv)
    conv = BatchNormalization(name=prefix + "_bn3")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv


def get_unet_resnet50(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output

    up6 = concatenate([(conv5), conv4], axis=-1)
    conv6 = decoder_block(up6, 1024, "conv6_1")

    up7 = concatenate([(conv6), conv3], axis=-1)
    conv7 = decoder_block(up7, 512, "conv7_1")

    up8 = concatenate([(conv7), conv2], axis=-1)
    conv8 = decoder_block(up8, 256, "conv8_1")

    up9 = concatenate([(conv8), conv1], axis=-1)
    conv9 = decoder_block(up9, 128, "conv9_1")

    up10 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name=prefix + "_Transp_conv")(conv9)
    conv10 = Conv2D(64, (3, 3), padding="same", strides=(1,1), name="conv11")(up10)
    conv11 = Conv2D(64, (2, 2), padding="same", strides=(1,1), name="conv11")(up10)
    # conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = models.Model(resnet_base.input, x)
    return model