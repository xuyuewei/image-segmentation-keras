import argparse
from tensorflow.python.keras import models

import tf_img_prepro_aug
import tensorflow as tf
tf.enable_eager_execution()
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--images_path", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_shape", type=int , default = [480,160])

args = parser.parse_args()

images_path = args.images_path
input_shape =  args.input_shape
output_path = args.output_path
save_weights_path = os.path.join(args.save_weights_path, 'weights.hdf5')

left_img_array = tf.data.Dataset.list_files(images_path+'/*10.png',shuffle=False)
right_img_array = tf.data.Dataset.list_files(images_path+'/*11.png',shuffle=False)
img_array = [[l,r] for l,r in zip(left_img_array,right_img_array)]

img_data = img_array.map(lambda x,y: tf_img_prepro_aug.load_stereo_jpeg(x,y,input_shape))
num_of_samples = len(img_array)

#load model
seg_depth_model = models.load_model(save_weights_path)

for i in range(num_of_samples):
    seg_depth = seg_depth_model.predict(img_data[i])
    cv2.imwrite(os.path.join(output_path,'seg' + str(i)), seg_depth[0])
    cv2.imwrite(os.path.join(output_path,'depth' + str(i)), seg_depth[1])

