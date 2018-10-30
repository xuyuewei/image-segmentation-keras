import tensorflow as tf
import tensorflow.contrib as tfcontrib
import numpy as np

def load_jpeg(image_path,resize = (256,256)):
	image = tf.read_file(image_path)
	image = tf.image.resize_images(image, resize)
  	image = tf.image.decode_jpeg(image)
  	image = tf.cast(image, tf.float32)
	return image

def load_stereo_jpeg(left_image_path,right_image_path,resize = (256,256)):
	left_image = tf.read_file(left_image_path)
	right_image_path = tf.read_file(right_image_path)
	left_image_path = tf.image.resize_images(left_image_path, resize)
    right_image_path = tf.image.resize_images(right_image_path, resize)
  	left_image = tf.image.decode_jpeg(left_image)
	right_image = tf.image.decode_jpeg(right_image)
  	left_image = tf.cast(left_image, tf.float32)
	right_image = tf.cast(right_image, tf.float32)
	return [left_image,right_image]


def train_val_split(img_array,ratio = 10):
    num_of_samples = len(img_array)
	random_index = tf.random_uniform(num_of_samples//ratio , minval = 0,maxval = num_of_samples,dtype = tf.int32)
    val_data = [img_array[x] for x in random_index]
    return val_data

def shift_img(input_imgs, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range:
        width_shift_range = tf.random_uniform([], -width_shift_range * img_shape[1],
                                                  width_shift_range * img_shape[1])
    if height_shift_range:
        height_shift_range = tf.random_uniform([],-height_shift_range * img_shape[0],
                                                   height_shift_range * img_shape[0])
      # Translate all
	for i in range(len(input_imgs)):
		input_imgs[i] = tfcontrib.image.translate(input_imgs[i],[width_shift_range, height_shift_range])
		
    return input_imgs
	
def ranflip_img(input_imgs, label_img,horizontal_flip, vertical_flip):
	if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        input_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
									   lambda: (tf.image.flip_left_right(input_img), tf.image.flip_left_right(label_img)),
									   lambda: (input_img, label_img))
	if vertical_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        input_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
									   lambda: (tf.image.flip_up_down(input_img), tf.image.flip_up_down(label_img)),
									   lambda: (input_img, label_img))
    return input_img, label_img

def flipran_img(input_imgs):
	flip_prob = tf.random_uniform([], 0.0, 1.0)
	for i in range(len(input_imgs)):
		input_imgs[i] = tf.cond(tf.less(flip_prob, 0.3),
							   lambda: (tf.image.flip_up_down(tf.image.flip_left_right(input_imgs[i]))
										lambda: (tf.cond(tf.less(flip_prob,0.6),
														 lambda:(tf.image.flip_up_down(input_imgs[i])),
														 lambda:(tf.image.flip_left_right(input_imgs[i]))))))
  
    return input_imgs

def ranrot_img(input_img, label_img,angle):
	if angle:
		rot_prob = tf.random_uniform([], 0.0, 1.0)
        input_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
									   lambda: (tf.contrib.image.rotate(input_img,angle), tf.contrib.image.rotate(label_img,angle)),
									   lambda: (input_img, label_img))
	return input_img, label_img

def rot_randomangle(input_imgs, angle):
	if angle:
		random_angle = tf.random_uniform([], 0.2, 1.0)*3.14*angle/180
		for i in range(len(input_imgs)):
        	input_imgs[i] = tf.contrib.image.rotate(input_imgs[i],angle)
	return input_imgs

def projective_random_transform(input_imgs,angle,img_size):
    if angle:
		random_angle = tf.random_uniform([], 0.5, 1.0)*angle
		transform = tf.contrib.image.angles_to_projective_transforms(angles,img_size[0],img_size[1])
		for i in range(len(input_imgs)):
			input_imgs[i] = tf.contrib.image.transform(input_imgs[i],transform)
    return input_imgs

def augmentation(input_imgs,
				 label_imgs,
				 scale = 1,  # Scale image e.g. 1 / 255.
				 hue_delta = 0.2,  # Adjust the hue of an RGB image by random factor
				 brightness = 0.3,
				 saturation = 0.2,
				 ranhorizontal_flip = False,  # Random left right flip,
				 ranvertical_flip = False,
				 ran_flip = False,
				 ranrot = False,
				 angle = 90,
				 projective_transform_angle = 90,
				 width_shift_range=0,  # Randomly translate the image horizontally
				 height_shift_range=0):  # Randomly translate the image vertically 
	
	num_of_parallel_imgs = len(input_imgs)
	for i in range(num_of_parallel_imgs):
	    input_imgs[i] = tf.image.random_hue(input_imgs[i], hue_delta) 
        input_imgs[i] = tf.image.random_brightness(input_imgs[i], brightness) 
        input_imgs[i] = tf.image.random_saturation(input_imgs[i], saturation) 

	#merge process
	input_imgs.extend(label_imgs)
	
	num_of_parallel_imgs = len(input_imgs)
	
	random_choice = tf.random_uniform([], 0.0, 1.0)
	
	for i in range(len(input_imgs)):
        input_imgs[i] = projective_random_transform(input_imgs[i],projective_transform_angle,resize)
		input_imgs[i] = shift_img(input_imgs[i], width_shift_range, height_shift_range)
		input_imgs[i] = tf.to_float(input_imgs[i]) * scale 
		
	if tf.less(random_choice, 0.5):
		for i in range(num_of_parallel_imgs):
			input_imgs[i] = flipran_img(input_imgs[i])
	else:
		for i in range(num_of_parallel_imgs):
			input_imgs[i] = rot_randomangle(input_imgs[i], angle)

	return input_imgs

