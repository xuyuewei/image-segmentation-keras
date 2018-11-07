import argparse
import tf_segdepth_model
import cv2 as cv
import tf_seg_stereo_loss
import tf_preprocess
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras import preprocessing
import os

tf.enable_eager_execution()


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--images_path", type = str  )
parser.add_argument("--seg_path", type = str  )
parser.add_argument("--depth_path", type = str  )
parser.add_argument("--input_shape", type=int , default = [480,160] )

parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--retrain", type = int, default = False )
parser.add_argument("--batch_size", type = int, default = 1 )
parser.add_argument("--validate", type = bool, default = False )
parser.add_argument("--val_batch_size", type = int, default = 1 )
parser.add_argument("--val_ratio", type = int, default = 0.1 )

args = parser.parse_args()

images_path = args.images_path
seg_path = args.seg_path
depth_path = args.depth_path
batch_size = args.batch_size
input_shape = args.input_shape
validate = args.validate
val_batch_size = args.val_batch_size
val_ratio = args.val_ratio

save_weights_path = os.path.join(args.save_weights_path, 'weights.hdf5')
epochs = args.epochs
retrain = args.retrain

left_img_array = [x for x in os.listdir(images_path)  if os.path.splitext(x)[0][-1]=='0']
right_img_array = [x for x in os.listdir(images_path)  if os.path.splitext(x)[0][-1]=='1']
seg_array = [x for x in os.listdir(images_path)]
depth_array = [x for x in os.listdir(images_path)]

num_of_train_samples = len(left_img_array)

#train_val_split
val_generator = None
if validate:
    val_left_img_array = left_img_array[:num_of_train_samples*val_ratio]
    val_right_img_array = right_img_array[:num_of_train_samples*val_ratio]
    val_seg_array = seg_array[:num_of_train_samples*val_ratio]
    val_depth_array = depth_array[:num_of_train_samples*val_ratio]
    
    left_img_array = left_img_array[num_of_train_samples*val_ratio:]
    right_img_array = right_img_array[num_of_train_samples*val_ratio:]
    seg_array = seg_array[num_of_train_samples*val_ratio:]
    depth_array = depth_array[num_of_train_samples*val_ratio:]
    
    
    val_left_img_array = map(lambda x:tf_preprocess.cvload_img(x,input_shape),val_left_img_array)
    val_right_img_array = map(lambda x:tf_preprocess.cvload_img(x,input_shape),val_right_img_array)
    val_seg_array = map(lambda x:tf_preprocess.cvload_img(x,input_shape),val_seg_array)
    val_depth_array = map(lambda x:tf_preprocess.cvload_img(x,input_shape),val_depth_array)
    
    val_generator = zip(val_left_img_array,val_right_img_array,val_seg_array,val_depth_array)

#data augmentation
left_img_array = map(lambda x:tf_preprocess.cvload_img(x,input_shape),left_img_array)
right_img_array = map(lambda x:tf_preprocess.cvload_img(x,input_shape),right_img_array)
seg_array = map(lambda x:tf_preprocess.cvload_img(x,input_shape),seg_array)
depth_array = map(lambda x:tf_preprocess.cvload_img(x,input_shape),depth_array)
                                   
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     brightness_range=0.3, 
                     shear_range=0.3, 
                     zoom_range=0.2,
                     horizontal_flip=True, 
                     vertical_flip=True)

left_img_datagen = preprocessing.image.ImageDataGenerator(**data_gen_args)
right_img_datagen = preprocessing.image.ImageDataGenerator(**data_gen_args)
seg_img_datagen = preprocessing.image.ImageDataGenerator(**data_gen_args)
depth_img_datagen = preprocessing.image.ImageDataGenerator(**data_gen_args)

seed = 1
left_img_datagen.fit(left_img_array, augment=True, seed=seed)
right_img_datagen.fit(right_img_array, augment=True, seed=seed)
seg_img_datagen.fit(seg_array, augment=True, seed=seed)
depth_img_datagen.fit(depth_array, augment=True, seed=seed)

left_imggen = left_img_datagen.flow(left_img_array,batch_size = batch_size)
right_imggen = left_img_datagen.flow(right_img_array,batch_size = batch_size)
seg_imggen = left_img_datagen.flow(seg_array,batch_size = batch_size)
depth_imggen = left_img_datagen.flow(depth_array,batch_size = batch_size)

train_generator = zip(left_imggen, right_imggen,seg_imggen,depth_imggen)


if retrain:
    #load model
    seg_depth_model = models.load_model(save_weights_path)
else:
    #create unet model
    segdep_model = tf_segdepth_model.segdepth()
                                            
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_weights_path, monitor='val_dice_loss', save_best_only=True, verbose=1)
EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_dice_loss', min_delta=0.01,patience=1,verbose=1)

#train
segdep_model.compile(optimizer="adam", loss= tf_seg_stereo_loss.cat_regression_loss, metrics= [tf_seg_stereo_loss.dice_loss,tf_seg_stereo_loss.smooth_l1])

history = segdep_model.fit_generator(train_generator, 
                                     epochs=epochs,
                                     validation_data=val_generator,
                                     callbacks=[ModelCheckpoint,EarlyStopping])
