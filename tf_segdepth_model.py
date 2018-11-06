import seg_stereo_loss

import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

import numpy as np

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
    decodercon = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decodercon)
    
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    #res
    decoder = layers.add([decodercon,decoder])
    
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
    
def spp_module(input_tensors):
    u1 = layers.UpSampling2D(size=(32, 32))(input_tensors)
    u1s = conv_block(u1,16)
    
    u2 = layers.UpSampling2D(size=(16, 16))(u1s)
    u2s = conv_block(u2,16)
    
    u3 = layers.UpSampling2D(size=(8, 8))(u2s)
    u3s = conv_block(u3,16)

    
    u4 = layers.UpSampling2D(size=(4, 4))(u3s)
    u4s = conv_block(u4,16)
    
    u5 = layers.UpSampling2D(size=(2, 2))(u4s)
    u5s = conv_block(u5,16)
    
    spp = layers.Concatenate([u1s,u2s,u3s,u4s,u5s],axis=-1)
    spp = conv_block(spp,32)
    return spp

def segdepth(img_shape = (256,256),loss = cat_regression_loss,optimizer='adam',metrics=[dice_loss,smooth_l1]):
    #make sure the img_shape can be devided by 2^8.(32)
    left_inputs = layers.Input(shape=img_shape)
    right_inputs = layers.Input(shape=img_shape)

    #left right images shared conv
    lres0,rres0 = res_shared_conv2(32,left_inputs,right_inputs)
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
    center1 = conv_block(lrcon4, 1024)
    
    #spp module
    spp0 = spp_module(center0)
    spp1 = spp_module(center1)
    
    #segment part
    #u-net
    sdecoder4 = res_convtrans_u_block(center0, lres4, 512)
    
    sdecoder3 = res_convtrans_u_block(sdecoder4, lres3, 256)
    
    sdecoder2 = res_convtrans_u_block(sdecoder3, lres2, 128)
    
    sdecoder1 = res_convtrans_u_block(sdecoder2, lres1, 64)
    
    sdecoder0 = res_convtrans_u_block(sdecoder1, lres0, 32)
    #merge u and spp
    u_spp0 = layers.Concatenate([spp0,sdecoder0],axis=-1)
    
    seg_outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(u_spp0)
    
    #depth estimation part
    #similar to unet
    ddecoder4 = res_convtrans_u_block(center1, lrcon4, 512)
    ddecoder41 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder4)
    ddecoder42 = convtrans_block(ddecoder41, ddecoder4, 256)
    
    ddecoder3 = res_convtrans_u_block(ddecoder42, lrcon3, 256)
    ddecoder31 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder3)
    ddecoder32 = res_convtrans_u_block(ddecoder31, ddecoder3, 128)
    
    ddecoder2 = res_convtrans_u_block(ddecoder32, lrcon2, 128)
    ddecoder21 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder2)
    ddecoder22 = res_convtrans_u_block(ddecoder21, ddecoder2, 64)
    
    ddecoder1 = convtrans_block(ddecoder22, lrcon1, 64)
    ddecoder11 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder1)
    ddecoder12 = res_convtrans_u_block(ddecoder11, ddecoder1, 64)
    
    ddecoder0 = res_convtrans_u_block(ddecoder12, lrcon0, 32)
    ddecoder01 = layers.MaxPooling2D((2, 2), strides=(2, 2))(ddecoder0)
    ddecoder02 = res_convtrans_u_block(ddecoder01, ddecoder0, 32)
    
    #merge u and spp
    u_spp1 = layers.Concatenate([spp1,ddecoder02],axis=-1)
    
    dep_outputs = layers.Conv2D(1, (1, 1), activation='softmax')(u_spp1)

    segdep_model = models.Model(inputs=[left_inputs,right_inputs], outputs=[seg_outputs,dep_outputs])
    
    segdep_model.summary()
    segdep_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return segdep_model
    
    
    
    
        
    
