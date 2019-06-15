import pandas as pd
import numpy as np
import os


from skimage.data import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.morphology import binary_opening, disk, label
from keras.preprocessing.image import ImageDataGenerator


def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift 
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks


def balancing_train(df, rate_of_has_ship, ship_dir_train):
    
    df_temp = df[["ImageId", "EncodedPixels"]].reset_index(drop=True)
    
    
    # some files are too small/corrupt
    df_temp['file_size_kb'] = df_temp['ImageId'].map(lambda c_img_id: 
                                                               os.stat(os.path.join(ship_dir_train, c_img_id)).st_size/1024)
    df_temp = df_temp[df_temp['file_size_kb'] > 80] # keep only +50kb files

    count_img_with_ships = len(df_temp[df_temp.EncodedPixels.notnull()])
    max_img_no_ship = round((1-rate_of_has_ship)*2*count_img_with_ships)
    
    #count_with_no_ship > count_with_ship
    #Take maximum of images on img_with_ship
    #Take maximum_img_no_ship on images data set (randomly choosen)
    if rate_of_has_ship != 0:
        balanced_train_df = df_temp[df_temp.EncodedPixels.notnull()]
        balanced_train_df = balanced_train_df.append(df_temp[df_temp.EncodedPixels.isna()].sample(max_img_no_ship), ignore_index=True)
    elif rate_of_has_ship == 1:
        balanced_train_df = df_temp
    elif rate_of_has_ship == 1:
        balanced_train_df = df_temp[df_temp.EncodedPixels.notnull()]

    df_train = balanced_train_df[["ImageId", "EncodedPixels"]].reset_index(drop=True)
    return df_train


def make_image_gen(in_df, train_image_dir, batch_size=48, img_scalling=None):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            if img_scalling is not None:
                c_img = c_img[::img_scalling[0], ::img_scalling[1]]
                c_mask = c_mask[::img_scalling[0], ::img_scalling[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []



def create_aug_gen(in_gen, image_gen, label_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)
        g_y = label_gen.flow(in_y, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)
        