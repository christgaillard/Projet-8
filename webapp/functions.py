import re
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.losses import binary_crossentropy
import cv2
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout, Reshape, UpSampling2D
from tensorflow.python.keras.utils.all_utils import Sequence
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.utils.all_utils import *
import segmentation_models as sm
from tensorflow.python.keras.saving.save import load_model

from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

import albumentations as A



def round_clip_0_1 ( x, **kwargs ) :
    return x.round().clip(0, 1)



def get_training_augmentation():
    '''
    Transforme les images et masks d'orrigine afin d'augmenter la quntité de features pour l'apprentissage.
    p définie une proba d'apliquer ou non l'augmentation
    :return: Image etmasque.
    '''
    train_transform = [

        A.HorizontalFlip(p=0.5),
        A.RandomCrop(width=224, height=160),
        A.OneOf(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=[-0.2, 0.2],
                    contrast_limit=0.2,
                    p=0.5,
                ),
                A.RandomGamma(p=0.5),
            ],
            p=0.8,
        )

    ]

    return A.Compose(train_transform)


def get_validation_augmentation (img_height,img_width) :
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(img_height, img_width)
    ]
    return A.Compose(test_transform)


def get_preprocessing ( preprocessing_fn ) :
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def Weighted_BCEnDice_loss ( y_true, y_pred ) :
    # if you are using this loss for multi-class segmentation then uncomment
    # following lines
    if y_pred.shape[-1] <= 1 :
        # #     # activate logits
        y_pred = tf.keras.activations.sigmoid(y_pred)
    elif y_pred.shape[-1] >= 2 :
        # #     # activate logits
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
    # #     # convert the tensor to one-hot for multi-class segmentation
    #      y_true = K.squeeze(y_true, 3)
    #      y_true = tf.cast(y_true, "int32")
    #      y_true = tf.one_hot(y_true, num_class, axis=-1)

    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
        y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_dice_loss(y_true, y_pred, weight) + weighted_bce_loss(y_true, y_pred, weight)
    return loss


def load_model(modelpath,weightspath):
    '''
    Charge le model et les weight ainsi que les custom objects utilisés pour l'apprentissage du model.
    :param modelpath: string
    :param weightspath: string
    :return: Object
    '''
    model = tf.keras.models.load_model(
        modelpath,
        custom_objects={'Weighted_BCEnDice_loss' : Weighted_BCEnDice_loss, 'iou_score' : sm.metrics.iou_score})
    # #model = sm.FPN('efficientnetb7', classes=8, activation='softmax', encoder_weights='imagenet')
    # #model = Unet([160, 224,3],8)
    # #model.compile(loss=sm.losses.cce_jaccard_loss,
    #               optimizer='adam',
    #               metrics=['accuracy', sm.metrics.iou_score])
    # #model_load_path = '/home/christophe/Documents/OpenClassRoom/Projet8/model/unet_30_epoc_categorical_crossentropy_with_augmentation.h5'
    model_load_path = '/home/christophe/Documents/OpenClassRoom/Projet8/model/FPN_pretrained_eficiannet_ecc_jaccard_loss.h5'
    model.load_weights(weightspath)
    return model


'''

'''
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        newimage = name+".png"
        plt.imshow(image)
        plt.savefig('/home/christophe/Documents/OpenClassRoom/Projet8/api/pythonProject/static/test/' + name)
    plt.show()
    return newimage

'''

'''
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


'''

'''
class DataGen(Sequence) :
    def __init__(self, image_file,mask_file,img_height=160,img_width=224, batch_size=1, shuffle=False, augmentation=None, preprocessing=None ) :
        self.image_file = image_file
        self.mask_file = mask_file
        self.image_height = img_height
        self.image_width = img_width
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.cats = {'void': [0, 1, 2, 3, 4, 5, 6],
                        'flat': [7, 8, 9, 10],
                        'construction': [11, 12, 13, 14, 15, 16],
                        'object': [17, 18, 19, 20],
                        'nature': [21, 22],
                        'sky': [23],
                        'human': [24, 25],
                        'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}



    def __getitem__ ( self, i ) :

        #start = i * self.batch_size
        #stop = (i + 1) * self.batch_size

        images, masks = [], []
        img = image.img_to_array(image.load_img(self.image_file, target_size=(self.image_height, self.image_width)))
        _mask = image.img_to_array(image.load_img(self.mask_file, color_mode="grayscale", target_size=(self.image_height, self.image_width)))

        labels = np.unique(_mask)
        _mask = np.squeeze(_mask)
        mask = np.zeros((_mask.shape[0], _mask.shape[1], 8))
        for i in range(-1, 34) :
            if i in self.cats['void'] :
                mask[:, :, 0] = np.logical_or(mask[:, :, 0], (_mask == i))
            elif i in self.cats['flat'] :
                mask[:, :, 1] = np.logical_or(mask[:, :, 1], (_mask == i))
            elif i in self.cats['construction'] :
                 mask[:, :, 2] = np.logical_or(mask[:, :, 2], (_mask == i))
            elif i in self.cats['object'] :
                mask[:, :, 3] = np.logical_or(mask[:, :, 3], (_mask == i))
            elif i in self.cats['nature'] :
                mask[:, :, 4] = np.logical_or(mask[:, :, 4], (_mask == i))
            elif i in self.cats['sky'] :
                mask[:, :, 5] = np.logical_or(mask[:, :, 5], (_mask == i))
            elif i in self.cats['human'] :
                mask[:, :, 6] = np.logical_or(mask[:, :, 6], (_mask == i))
            elif i in self.cats['vehicle'] :
                mask[:, :, 7] = np.logical_or(mask[:, :, 7], (_mask == i))

         # apply augmentations
        if self.augmentation :
            sample = self.augmentation(image=img, mask=mask)
            img, _mask = sample['image'], sample['mask']

            # apply preprocessing
        if self.preprocessing :
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        masks.append(mask)
        images.append(img)

        mask_batch = np.stack(masks, axis=0)
        image_batch = np.stack(images, axis=0)
        # print(image_batch.shape,'---',mask_batch.shape)

        return image_batch, mask_batch



def Unet(input_size, n_classes=8):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu',
                   padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(n_classes, 1, activation = 'softmax')(conv9)

    return Model(inputs, conv10)