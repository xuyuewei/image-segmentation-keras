import tensorflow as tf
import tensorflow.contrib as tfcontrib

def load_jpeg(image_path):
	image = tf.read_file(image_path)
  	image = tf.image.decode_jpeg(image)
  	image = tf.cast(image, tf.float32)
	return image

def shift_img(input_img, label_img, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range:
        width_shift_range = tf.random_uniform([], -width_shift_range * img_shape[1],
                                                  width_shift_range * img_shape[1])
    if height_shift_range:
        height_shift_range = tf.random_uniform([],-height_shift_range * img_shape[0],
                                                   height_shift_range * img_shape[0])
      # Translate both 
    input_img = tfcontrib.image.translate(input_img,[width_shift_range, height_shift_range])
    label_img = tfcontrib.image.translate(label_img,[width_shift_range, height_shift_range])
    return input_img, label_img
	
def ranflip_img(input_img, label_img,horizontal_flip, vertical_flip):
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

def ranrot_img(input_img, label_img,angle):
	if angle:
		rot_prob = tf.random_uniform([], 0.0, 1.0)
        input_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
									   lambda: (tf.contrib.image.rotate(input_img,angle), tf.contrib.image.rotate(label_img,angle)),
									   lambda: (input_img, label_img))
	return input_img, label_img

def augmentation(input_img,
				 label_img,
				 resize = [128, 256], # Resize the image to some size e.g. [256, 256]
				 scale = 1,  # Scale image e.g. 1 / 255.
				 hue_delta = 0,  # Adjust the hue of an RGB image by random factor
				 brightness = 0,
				 saturation = 0,
				 horizontal_flip = False,  # Random left right flip,
				 vertical_flip = False,
				 width_shift_range=0,  # Randomly translate the image horizontally
				 height_shift_range=0):  # Randomly translate the image vertically 
	
	if resize is not None:
        # Resize both images
        label_img = tf.image.resize_images(label_img, resize)
        input_img = tf.image.resize_images(input_img, resize)
		
	if hue_delta:
        input_img = tf.image.random_hue(input_img, hue_delta) 
	
	if brightness:
        input_img = tf.image.random_brightness(input_img, brightness) 
		
	if saturation:
        input_img = tf.image.random_saturation(input_img, saturation) 

	input_img, label_img = ranflip_img(input_img, label_img, horizontal_flip, vertical_flip)
    input_img, label_img = shift_img(input_img, label_img, width_shift_range, height_shift_range)
	input_img, label_img = ranrot_img(input_img, label_img, angle)
    label_img = tf.to_float(label_img) * scale
    input_img = tf.to_float(input_img) * scale 
	
	return input_img, label_img
