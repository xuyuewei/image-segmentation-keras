import argparse
import tf_segdepth_model
import seg_stereo_loss
import tf_img_prepro_aug
import numpy as np
import tensorflow as tf
import os


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--images_path", type = str  )
parser.add_argument("--seg_path", type = str  )
parser.add_argument("--depth_path", type = str  )
parser.add_argument("--input_shape", type=int , default = [480,160] )

parser.add_argument("--epochs", type = int, default = 5 )
parser.add_argument("--retrain", type = int, default = False )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--val_batch_size", type = int, default = 2 )
parser.add_argument("--load_weights_path", type = str , default = "")

args = parser.parse_args()

images_path = args.images_path
seg_path = args.seg_path
depth_path = args.depth_path
batch_size = args.batch_size
input_shape = args.input_shape
validate = args.validate
save_weights_path = os.path.join(args.save_weights_path, 'weights.hdf5')
epochs = args.epochs
retrain = args.retrain
load_weights_path = args.load_weights_path

left_img_array = tf.data.Dataset.list_files(images_path+'/*10.png',shuffle=False)
right_img_array = tf.data.Dataset.list_files(images_path+'/*11.png',shuffle=False)
img_array = [[l,r] for l,r in zip(left_img_array,right_img_array)]

seg_array = tf.data.Dataset.list_files(seg_path,shuffle=False)
depth_array = tf.data.Dataset.list_files(depth_path,shuffle=False)
labels_array = [[l,r] for l,r in zip(seg_array,depth_array)]

img_labels = tf.data.Dataset.from_tensor_slices((img_array,labels_array))
img_labels = img_labels.shuffle(num_of_samples)

#train_val_split
num_of_samples = len(img_array)
val_labels_data = None
if validate:
    val_array = img_labels[:num_of_samples//10]
    img_labels = img_labels[num_of_samples//10:]
    val_data = val_array.map(lambda x: ([load_stereo_jpeg(x[0][0],x[0][1],input_shape),
                                       load_stereo_jpeg(x[1][0],x[1][1],input_shape)]))
    
    val_data = val_data.batch(batch_size)

#data augmentation
img_labels_data = img_labels.map(lambda x: ([load_stereo_jpeg(x[0][0],x[0][1],input_shape),
                                           load_stereo_jpeg(x[1][0],x[1][1],input_shape)]))
                                   
aug_train_data = img_labels_data.map(lambda x:augmentation(x,scale = 1/255))
img_labels_data = img_labels_data.concatenate(aug_train_data)
img_labels_data = img_labels_data.batch(batch_size)
num_of_train_samples = len(img_labels_data)

if retrain:
    #load model
    seg_depth_model = models.load_model(save_weights_path)
    segdep_model.compile(optimizer=optimizer, loss= categorical_regression, metrics= [dice_loss,smooth_l1])
else:
    #create unet model
    segdep_model = segdepth()
                                            
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_weights_path, monitor='val_dice_loss', save_best_only=True, verbose=1)
EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_dice_loss', min_delta=0.01,patience=1,verbose=1)

#train
history = segdep_model.fit(img_labels_data, 
                           steps_per_epoch=int(np.ceil(num_of_train_samples / float(batch_size))),
                           epochs=epochs,
                           validation_data=val_data,
                           validation_steps=int(np.ceil(num_of_train_samples / float(batch_size))),
                           callbacks=[ModelCheckpoint,EarlyStopping])
