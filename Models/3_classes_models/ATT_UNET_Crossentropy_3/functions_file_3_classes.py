# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:25:38 2022

@author: franc
"""

"""
@author: franc
"""
import os
import math
import random
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from keras import backend as K
#from keras.metrics import MeanIoU
from tensorflow.keras import models, layers
#from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout, add, Activation, UpSampling2D, multiply, BatchNormalization, Lambda, Input 
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from tensorflow.keras.metrics import Metric

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

NUM_CLASSES = 3

matplotlib.use('agg')

def normalize(input_mask):
    input_mask = np.where(input_mask == 127, 127.5, input_mask)
    input_mask = ((input_mask/255)*2)
    return input_mask

def plot_image(img):
    imshow(np.squeeze(img))
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def unet_model():
    start_neurons = 16
    FILTER_NUM = 16 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    #UP_SAMP_SIZE = 2
    dropout_rate = 0.1
    batch_norm = True
    input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
    inputs = inputs = layers.Input(input_shape, dtype=tf.float32)

    s = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)

    #conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(s)
    conv1 = conv_block(s, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    #conv1 = Dropout(0.1)(conv1)
    #conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    #conv1 = conv_block(conv1, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    #conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = conv_block(pool1, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    #conv2 = Dropout(0.1)(conv2)
    #conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    #conv2 = conv_block(conv2, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    #conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = conv_block(pool2, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    #conv3 = Dropout(0.2)(conv3)
    #conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    #conv3 = conv_block(conv3, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    
    #conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = conv_block(pool3, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    #conv4 = Dropout(0.2)(conv4)
    #conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    #conv4 = conv_block(conv4, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool4 = MaxPooling2D((2, 2))(conv4)
    
    
    # Middle
    #convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = conv_block(pool4, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)
    #convm = Dropout(0.3)(convm)
    #convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    #convm = conv_block(convm, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
        
    #uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = conv_block(uconv4, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    #uconv4 = Dropout(0.2)(uconv4)
    #uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    #uconv4 = conv_block(uconv4, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    #uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = conv_block(uconv3, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    #uconv3 = Dropout(0.2)(uconv3)
    #uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    #uconv3 = conv_block(uconv3, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    #uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = conv_block(uconv2, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    #uconv2 = Dropout(0.1)(uconv2)
    #uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    #uconv2 = conv_block(uconv2, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    #uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = conv_block(uconv1, FILTER_SIZE, 1*FILTER_NUM, dropout_rate, batch_norm)
    #uconv1 = Dropout(0.1)(uconv1)
    #uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    #uconv1 = conv_block(uconv1, FILTER_SIZE, 1*FILTER_NUM, dropout_rate, batch_norm)
        
    output_layer = Conv2D(NUM_CLASSES, (1,1), padding="same", activation="softmax")(uconv1)
    return tf.keras.Model(inputs=inputs, outputs=output_layer)


#####ATTENTION#####
def Attention_UNet(NUM_CLASSES=1, dropout_rate=0.1, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 16 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
    
    inputs = layers.Input(input_shape, dtype=tf.float32)
    s = tf.keras.layers.Lambda(lambda x: x / 255.0)(inputs)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    
    conv_128 = conv_block(s, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    #conv_64 = conv_block(inputs, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    
    # 1*1 convolutional layers
    #conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_64)
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('softmax')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="Attention_UNet")
    return model

def conv_block(x, filter_size, size, dropout, batch_norm=False):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    return conv

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)

# Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)

# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)

def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x
#####NOVODICE#####
    
class ConfusionMatrix(Metric):
    def __init__(self, num_classes, **kwargs):
        super(ConfusionMatrix, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(name='cm', shape=(num_classes, num_classes), initializer=tf.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.argmax(y_true, axis=-1)
        
        y_pred = tf.reshape(y_pred, shape=(-1,))
        y_true = tf.reshape(y_true, shape=(-1,))
        
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes)
        confusion_matrix = tf.cast(confusion_matrix, dtype=tf.float32) # cast to float32
        self.confusion_matrix.assign_add(confusion_matrix)

    def result(self):
        return tf.identity(self.confusion_matrix)

    def reset_state(self):
        tf.keras.backend.set_value(self.confusion_matrix, tf.zeros((self.num_classes, self.num_classes)))

    
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        cm = self.model.metrics[1]
        print("Confusion matrix: ")
        print(cm.result().numpy())   
        
def metrics_calc(tp, fp, fn, tn):
    
    acc = (tn + tp)/(tn + tp + fn + fp)
    dice = (2*tp)/(2*tp+fp+fn)
    iou = (tp)/(tp+fp+fn)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    
    acc = isNaN(acc)
    dice = isNaN(dice)
    iou = isNaN(iou)
    precision = isNaN(precision)
    recall = isNaN(recall)
    
    return(acc,dice,iou, precision, recall)

def metrics_class(cm):
#[HYPER] MUDAR PARA 3 CLASSES 
    class0_tp = cm[0,0]
    class0_fp = cm[0,1]+cm[0,2]
    class0_fn = cm[1,0]+cm[2,0]
    
    class1_tp = cm[1,1]
    class1_fp = cm[1,0]+cm[1,2]
    class1_fn = cm[0,1]+cm[2,1]
    
    class2_tp = cm[2,2]
    class2_fp = cm[2,0]+cm[2,1]
    class2_fn = cm[0,2]+cm[1,2]
    
    
    precision_class0 = (class0_tp)/(class0_tp+class0_fp)
    precision_class1 = (class1_tp)/(class1_tp+class1_fp)
    precision_class2 = (class2_tp)/(class2_tp+class2_fp)
    precision_class0 = isNaN(precision_class0)
    precision_class1 = isNaN(precision_class1)
    precision_class2 = isNaN(precision_class2)
    precision_mean = (precision_class0 + precision_class1 + precision_class2)/3
    
    precision_class0 = round(float(precision_class0),6)
    precision_class1 = round(float(precision_class1),6)
    precision_class2 = round(float(precision_class2),6)
    precision_mean = round(float(precision_mean),6)
    
    precision = [precision_mean, precision_class0, precision_class1, precision_class2]
    
    recall_class0 = (class0_tp)/(class0_tp+class0_fn)
    recall_class1 = (class1_tp)/(class1_tp+class1_fn)
    recall_class2 = (class2_tp)/(class2_tp+class2_fn)
    recall_class0 = isNaN(recall_class0)
    recall_class1 = isNaN(recall_class1)
    recall_class2 = isNaN(recall_class2)
    recall_mean = (recall_class0 + recall_class1 + recall_class2)/3
    
    recall_class0 = round(float(recall_class0),6)
    recall_class1 = round(float(recall_class1),6)
    recall_class2 = round(float(recall_class2),6)
    recall_mean = round(float(recall_mean),6)
    
    recall = [recall_mean, recall_class0, recall_class1, recall_class2]
    
    dice_class0 = (2*class0_tp)/(2*class0_tp+class0_fn+class0_fp)
    dice_class1 = (2*class1_tp)/(2*class1_tp+class1_fn+class1_fp)
    dice_class2 = (2*class2_tp)/(2*class2_tp+class2_fn+class2_fp)
    dice_class0 = isNaN(dice_class0)
    dice_class1 = isNaN(dice_class1)
    dice_class2 = isNaN(dice_class2)
    dice_mean = (dice_class0 + dice_class1 + dice_class2)/3
    
    dice_class0 = round(float(dice_class0),6)
    dice_class1 = round(float(dice_class1),6)
    dice_class2 = round(float(dice_class2),6)
    dice_mean = round(float(dice_mean),6)
    
    dice = [dice_mean, dice_class0, dice_class1, dice_class2]
    
    iou_class0 = (class0_tp)/(class0_tp+class0_fn+class0_fp)
    iou_class1 = (class1_tp)/(class1_tp+class1_fn+class1_fp)
    iou_class2 = (class2_tp)/(class2_tp+class2_fn+class2_fp)
    iou_class0 = isNaN(iou_class0)
    iou_class1 = isNaN(iou_class1)
    iou_class2 = isNaN(iou_class2)
    iou_mean = (iou_class0 + iou_class1 + iou_class2)/3
    
    iou_class0 = round(float(iou_class0),6)
    iou_class1 = round(float(iou_class1),6)
    iou_class2 = round(float(iou_class2),6)
    iou_mean = round(float(iou_mean),6)
    
    iou = [iou_mean, iou_class0, iou_class1, iou_class2]
    
    
    return (dice, iou, precision, recall)

    '''
    tn = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]
    tp = cm[1,1]
    
    acc = (tn + tp)/(tn + tp + fn + fp)
    dice = (2*tp)/(2*tp+fp+fn)
    iou = (tp)/(tp+fp+fn)
    precision = (tp)/(tp+fp)
    recall = (tp)/(tp+fn)
    acc = isNaN(acc)
    dice = isNaN(dice)
    iou = isNaN(iou)
    precision = isNaN(precision)
    recall = isNaN(recall)
    
    return(tn, fp, fn, tp, acc,dice,iou, precision, recall)
    '''
    
def isNaN(num):
    State = (num!= num)
    if (State == True):
        num = 0
    else:
        num = num
    return num
#####NOVO#########

def load_image(TRAIN_PATH, _id):
    img = imread(TRAIN_PATH + _id)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True, anti_aliasing=False)
    return img

def load_mask(TRAIN_MASK_PATH, _id):
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    mask_ = imread(TRAIN_MASK_PATH + _id)[:,:,]
    mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH,1), mode='constant', preserve_range=True, anti_aliasing=False,order=0)
    mask = np.maximum(mask, mask_)
    mask = normalize(mask)
    return mask

def step_decay(epoch):
    initial_lrate = 1e-3
    drop = 0.1
    epochs_drop = 15
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))    
    return lrate


def Plot_result(X_test, Y_test, Y_pred_img, img_pred_path):
    
    test_img = X_test[0]
    ground_truth=Y_test[0]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,0], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap='gray')
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(Y_pred_img, cmap='gray')
    plt.savefig(img_pred_path)
    #plt.show(block=False)
    #plt.pause(3)
    plt.close()
 

def Plot_graph(loss, graph_pred_path, labelx):
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'r')
    plt.title(labelx)
    plt.xlabel('Epochs')
    plt.ylabel(labelx.replace('Training',''))
    #plt.legend()
    plt.savefig(graph_pred_path)
    #plt.show()
    #plt.pause(3)
    plt.close()
    
def Plot_multiple(loss, loss1,loss2,loss3, graph_pred_path, label):
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'r', label=(label.replace('Training','')))
    plt.plot(epochs, loss1, 'g', label=(label.replace('Training','')+'class_1'))
    plt.plot(epochs, loss2, 'b', label=(label.replace('Training','')+'class_2'))
    plt.plot(epochs, loss3, 'y', label=(label.replace('Training','')+'class_3'))
    plt.title(label)
    plt.xlabel('Epochs')
    plt.ylabel(label.replace('Training',''))
    plt.legend()
    plt.savefig(graph_pred_path)
    #plt.show()
    #plt.pause(3)
    plt.close()


def create_folder(folder_name):
    
    folder_path =os.path.join(os.getcwd(), folder_name)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    
    return (folder_path)

def Generate_aug(images_to_generate, images_path, train_ids, masks_path, train_mask_ids, img_augmented_path, msk_augmented_path):
    
    images=[] # to store paths of images from folder
    masks=[]
    
    for im in train_ids:  # read image name from folder and append its path into "images" array     
        images.append(os.path.join(images_path,im))

    for msk in train_mask_ids:  # read image name from folder and append its path into "masks" array     
        masks.append(os.path.join(masks_path,msk))


    aug = A.Compose([
            A.HorizontalFlip (p=0.5),              
            A.Rotate(limit=[-5,5],p=0.5),
            A.ShiftScaleRotate (p=0.5, shift_limit=0, scale_limit=0, rotate_limit=0, shift_limit_x = [-0.07,0.07], shift_limit_y = [-0.03,0.03]), 
            A.RandomBrightnessContrast(p=0.6, contrast_limit=[-0.1,0.4], brightness_limit=[-0.1,0.4] ),
            ]
        )
    i=1
    while i<=images_to_generate: 
        number = random.randint(0, len(images)-1)  #PIck a number to select an image & mask
        image = images[number]
        mask = masks[number]
        print(image, mask)
        
        #image=random.choice(images) #Randomly select an image name
        original_image = imread(image)
        original_mask = imread(mask)
    
        augmented = aug(image=original_image, mask=original_mask)
        transformed_image = augmented['image']
        transformed_mask = augmented['mask']

        
        new_image_path= "%s/augmented_image_%s.png" %(img_augmented_path, i)
        new_mask_path = "%s/augmented_mask_%s.png" %(msk_augmented_path, i)
        imsave(new_image_path, transformed_image)
        imsave(new_mask_path, transformed_mask)
        i =i+1

