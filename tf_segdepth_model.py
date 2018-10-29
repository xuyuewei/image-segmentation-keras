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
    
def segdepth(img_shape = (256,256),loss = bce_dice_loss,optimizer='adam',metrics=[dice_loss]):
    inputs = layers.Input(shape=img_shape)
    linputs = inputs[:,:,:3]
    rinputs = inputs[:,:,3:]

    #left right images shared conv
    lres0,rres0 = res_shared_conv2(32,linputs,rinputs)
    lresp0 = layers.MaxPooling2D((2, 2), strides=(2, 2))(lres0)
    rresp0 = layers.MaxPooling2D((2, 2), strides=(2, 2))(rres0)
    #left right featuremaps merge
    lrres0 = layers.add([lresp0,rresp0])
    lrcon0 = layers.Concatenate([lresp0,rresp0,lrres0],axis=-1)
    
    lres1,rres1 = res_shared_conv2(64,lresp0,rresp0)
    lresp1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(lres1)
    rresp1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(rres1)
    #left right featuremaps merge
    lrres1 = layers.add([lresp1,rresp1])
    lrcon1 = layers.Concatenate([lresp1,rresp1,lrres1],axis=-1)
    
    lres2,rres2 = res_shared_conv2(128,lresp1,rresp1)
    lresp2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(lres2)
    rresp2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(rres2)
    #left right featuremaps merge
    lrres2 = layers.add([lresp2,rresp2])
    lrcon2 = layers.Concatenate([lresp2,rresp2,lrres2],axis=-1)
    
    lres3,rres3 = res_shared_conv2(256,lresp2,rresp2)
    lresp3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(lres3)
    rresp3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(rres3)
    #left right featuremaps merge
    lrres3 = layers.add([lresp3,rresp3]) 
    lrcon3 = layers.Concatenate([lresp3,rresp3,lrres3],axis=-1)
    
    lres4,rres4 = res_shared_conv2(512,lresp3,rresp3)
    lresp4 = layers.MaxPooling2D((2, 2), strides=(2, 2))(lres4)
    rresp4 = layers.MaxPooling2D((2, 2), strides=(2, 2))(rres4)
    #left right featuremaps merge
    lrres4 = layers.add([lresp4,rresp4]) 
    lrcon4 = layers.Concatenate([lresp4,rresp4,lrres4],axis=-1)
    
    #come to center
    center0 = conv_block(lresp4, 1024)
    center1 = encoder_block(lrcon4, 1024)
    
    #segment part
    sdecoder4 = res_convtrans_u_block(center0, lres4, 512)
    
    sdecoder3 = res_convtrans_u_block(sdecoder4, lres3, 256)
    
    sdecoder2 = res_convtrans_u_block(sdecoder3, lres2, 128)
    
    sdecoder1 = res_convtrans_u_block(sdecoder2, lres1, 64)
    
    sdecoder0 = res_convtrans_u_block(sdecoder1, lres0, 32)
    
    seg_outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(decoder0)
    
    #depth estimation part
    ddecoder4 = res_convtrans_u_block(center1, lrcon4, 512)
    ddecoder41 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder4)
    ddecoder42 = res_convtrans_u_block(ddecoder41, ddecoder4, 256)
    ddecoder43 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder42)
    ddecoder44 = convtrans_block(ddecoder43, ddecoder42, 512)
    
    ddecoder3 = res_convtrans_u_block(ddecoder44, lrcon3, 256)
    ddecoder31 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder3)
    ddecoder32 = res_convtrans_u_block(ddecoder31, ddecoder3, 128)
    ddecoder33 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder32)
    ddecoder34 = res_convtrans_u_block(ddecoder33, ddecoder32, 256)
    
    ddecoder2 = res_convtrans_u_block(ddecoder34, lrcon2, 128)
    ddecoder21 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder2)
    ddecoder22 = res_convtrans_u_block(ddecoder21, ddecoder2, 64)
    ddecoder23 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder22)
    ddecoder24 = res_convtrans_u_block(ddecoder23, ddecoder22, 128)
    
    ddecoder1 = convtrans_block(ddecoder2, lrcon1, 64)
    ddecoder11 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder1)
    ddecoder12 = res_convtrans_u_block(ddecoder11, ddecoder1, 32)
    ddecoder13 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder12)
    ddecoder14 = res_convtrans_u_block(ddecoder13, ddecoder12, 64)
    
    ddecoder0 = res_convtrans_u_block(ddecoder14, lrcon0, 32)
    ddecoder01 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder0)
    ddecoder02 = res_convtrans_u_block(ddecoder01, ddecoder0, 16)
    ddecoder03 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder02)
    ddecoder04 = res_convtrans_u_block(ddecoder03, ddecoder02, 32)
    
    dep_outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(ddecoder04)
    
    outputs = layers.Concatenate([seg_outputs,dep_outputs],axis=-1)
    
    segdep_model = models.Model(inputs=[inputs], outputs=[outputs])
    
    segdep_model.summary()
    segdep_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return segdep_model
    
    
    
    
        
    
