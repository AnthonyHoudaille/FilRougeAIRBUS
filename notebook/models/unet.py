from keras import models
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout 
from keras.layers import BatchNormalization, UpSampling2D, Dropout, AvgPool2D, SpatialDropout2D

def get_unet(input_shape=(256,256,3), UPSAMPLE_MODE='DECONV', NET_SCALING=None):

    # Build U-Net model
    def upsample_conv(filters, kernel_size, strides, padding):
        return Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
    def upsample_simple(filters, kernel_size, strides, padding):
        return UpSampling2D(strides)

    if UPSAMPLE_MODE=='DECONV':
        upsample=upsample_conv
    else:
        upsample=upsample_simple

    input_img = Input(input_shape, name = 'RGB_Input')
    pp_in_layer = input_img
    
    if NET_SCALING is not None:
        pp_in_layer = AvgPool2D(NET_SCALING)(pp_in_layer)
    
    pp_in_layer = BatchNormalization()(pp_in_layer)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (pp_in_layer)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)


    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = upsample(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = upsample(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    if NET_SCALING is not None:
        outputs = UpSampling2D(NET_SCALING) (outputs)
    

    model = models.Model(inputs=[input_img], outputs=[outputs])

    return model