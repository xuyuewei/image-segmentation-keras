import bce_dice_loss

import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

import numpy as np

#selfmade_shallow part of the resnet_50

def conv_block(input_tensor, num_filters):
    #tf implemention
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder
    
def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder
    
def res_convtrans_u_block(input_tensor, concat_tensor, num_filters):
    #convtrans
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    #U_block
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    #res
    decoder1 = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.add([decoder1,decoder])
    
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder

def res_shared_conv2(filters,linput_tensors,rinput_tensors):
    sconv0 = layers.Conv2D(filters, (3, 3), padding='same')
    sconv00 = layers.Conv2D(filters, (3, 3), padding='same')
    #
    lsconv0 = sconv0(linput_tensors)
    rsconv0 = sconv0(rinput_tensors)
    
    lbn0 = layers.BatchNormalization()(lsconv0)
    rbn0 = layers.BatchNormalization()(rsconv0)
    
    lrelu0 = layers.Activation('relu')(lbn0)
    rrelu0 = layers.Activation('relu')(rbn0)
    
    lsconv1 = sconv00(lrelu0)
    rsconv1 = sconv00(rrelu0)
    
    lres0 = layers.add([linput_tensors,lsconv1])
    rres0 = layers.add([rinput_tensors,rsconv1])
    
    lbn1 = layers.BatchNormalization()(lres0)
    rbn1 = layers.BatchNormalization()(rres0)
    
    lrelu1 = layers.Activation('relu')(lbn1)
    rrelu1 = layers.Activation('relu')(rbn1)
    
    return lrelu1,rrelu1
    
def segdepth(img_shape = (256,512),loss = bce_dice_loss,optimizer='adam',metrics=[dice_loss]):
    inputs = layers.Input(shape=img_shape)
    linputs = inputs[:,:img_shape//2]
    rinputs = inputs[:,img_shape//2:]

    #left right images shared conv
    lres0,rres0 = res_shared_conv2(32,linputs,rinputs)
    lresp0 = layers.MaxPooling2D((2, 2), strides=(2, 2))(lres0)
    rresp0 = layers.MaxPooling2D((2, 2), strides=(2, 2))(rres0)
    #left right featuremaps merge
    lrres0 = layers.add([lresp0,rresp0])
    lrcon0 = layers.Concatenate(axis=-1)([lresp0,rresp0])
    
    lres1,rres1 = res_shared_conv2(64,lresp0,rresp0)
    lresp1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(lres1)
    rresp1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(rres1)
    #left right featuremaps merge
    lrres1 = layers.add([lresp1,rresp1])
    lrcon1 = layers.Concatenate(axis=-1)([lresp1,rresp1])
    
    lres2,rres2 = res_shared_conv2(128,lresp1,rresp1)
    lresp2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(lres2)
    rresp2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(rres2)
    #left right featuremaps merge
    lrres2 = layers.add([lresp2,rresp2])
    lrcon2 = layers.Concatenate(axis=-1)([lresp2,rresp2])
    
    lres3,rres3 = res_shared_conv2(256,lresp2,rresp2)
    lresp3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(lres3)
    rresp3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(rres3)
    #left right featuremaps merge
    lrres3 = layers.add([lresp3,rresp3]) 
    lrcon3 = layers.Concatenate(axis=-1)([lresp3,rresp3])
    
    #come to center
    center = conv_block(lresp3, 512)
    
    #segment part
    decoder3 = convtrans_block(center, lres3, 256)
    decoder2 = convtrans_block(decoder3, lres2, 128)
    decoder1 = convtrans_block(decoder2, lres1, 64)
    decoder0 = convtrans_block(decoder1, lres0, 32)
    seg_outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(decoder0)
    
    #depth estimation part
    
    
    
    
    
    
    
    
        
    
