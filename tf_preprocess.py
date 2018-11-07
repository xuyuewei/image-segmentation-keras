import tensorflow as tf
import tensorflow.contrib as tfcontrib

def load_jpeg(image_path,resize = (256,256)):
    image = tf.read_file(image_path)
    
    image = tf.image.decode_jpeg(image)
    
    image = tf.image.resize_images(image, resize)

    image = tf.cast(image, tf.float32)
    return image

def load_stereo_jpeg(left_image_path,right_image_path,resize = (256,256)):
    left_image = tf.read_file(left_image_path)
    right_image = tf.read_file(right_image_path)
    
    left_image = tf.image.decode_jpeg(left_image)
    right_image = tf.image.decode_jpeg(right_image)
    
    left_image = tf.image.resize_images(left_image, resize)
    right_image = tf.image.resize_images(right_image, resize)

    left_image = tf.cast(left_image, tf.float32)
    right_image = tf.cast(right_image, tf.float32)
    return [left_image,right_image]

def train_val_split(img_array,ratio = 10):
    num_of_samples = len(img_array)
    random_index = tf.random_uniform(num_of_samples//ratio , minval = 0,maxval = num_of_samples,dtype = tf.int32)
    val_data = [img_array[x] for x in random_index]
    return val_data

def shift_img(input_imgs,label_imgs, width_shift_range, height_shift_range,img_size):
    """This fn will perform the horizontal or vertical shift"""
    img_shape = img_size
    if width_shift_range:
        width_shift_range = tf.random_uniform([], -width_shift_range * img_shape[1],
                                              width_shift_range * img_shape[1])
        if height_shift_range:
            height_shift_range = tf.random_uniform([],-height_shift_range * img_shape[0],
                                                   height_shift_range * img_shape[0])
      # Translate all
    input_imgs = tfcontrib.image.translate(input_imgs,[width_shift_range, height_shift_range])
    label_imgs = tfcontrib.image.translate(label_imgs,[width_shift_range, height_shift_range])
        
    return input_imgs
	
def ranflip_img(input_imgs, label_imgs,horizontal_flip, vertical_flip):
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
    return [input_img, label_img]

def flipran_img(input_imgs,label_imgs):
    flip_prob = tf.random_uniform([], 0.0, 1.0)
    input_imgs= tf.cond(tf.less(flip_prob, 0.3),
                        lambda: (tf.image.flip_up_down(tf.image.flip_left_right(input_imgs)),
                                 lambda: (tf.cond(tf.less(flip_prob,0.6),
                                                  lambda:(tf.image.flip_up_down(input_imgs)),
                                                  lambda:(tf.image.flip_left_right(input_imgs))))))
    label_imgs= tf.cond(tf.less(flip_prob, 0.3),
                        lambda: (tf.image.flip_up_down(tf.image.flip_left_right(label_imgs)),
                                 lambda: (tf.cond(tf.less(flip_prob,0.6),
                                                  lambda:(tf.image.flip_up_down(label_imgs)),
                                                  lambda:(tf.image.flip_left_right(label_imgs))))))
    return input_imgs,label_imgs

def ranrot_img(input_img, label_img,angle):
    if angle:
        rot_prob = tf.random_uniform([], 0.0, 1.0)
        input_img, label_img = tf.cond(tf.less(rot_prob, 0.5),
                                       lambda: (tf.contrib.image.rotate(input_img,angle), tf.contrib.image.rotate(label_img,angle)),
                                       lambda: (input_img, label_img))
    return [input_img, label_img]

def rot_randomangle(input_imgs,label_imgs, angle):
    if angle:
        random_angle = tf.random_uniform([], 0.2, 1.0)*3.14*angle/180
        input_imgs = tf.contrib.image.rotate(input_imgs,random_angle)
        label_imgs = tf.contrib.image.rotate(label_imgs,random_angle)
    return input_imgs,label_imgs

def projective_random_transform(input_imgs,label_imgs,angle,img_size):
    if angle:
        random_angle = tf.random_uniform([], 0.5, 1.0)*angle
        transform = tf.contrib.image.angles_to_projective_transforms(random_angle,img_size[0],img_size[1])
        input_imgs = tf.contrib.image.transform(input_imgs,transform)
        label_imgs = tf.contrib.image.transform(label_imgs,transform)
        
    return input_imgs,label_imgs

def augmentation(input_imgs,
                 label_imgs,
                 scale = 1,  # Scale image e.g. 1 / 255.
                 hue_delta = 0.2,  # Adjust the hue of an RGB image by random factor
                 brightness = 0.3,
                 lsaturation = 0.1,
                 usaturation = 0.3,
                 ranhorizontal_flip = False,  # Random left right flip,
                 ranvertical_flip = False,
                 ran_flip = False,
                 ranrot = False,
                 angle = 90,
                 projective_transform_angle = 90,
                 img_size = [480,160],
                 width_shift_range=0,  # Randomly translate the image horizontally
                 height_shift_range=0):  # Randomly translate the image vertically
    
    
    input_imgs = tf.image.random_hue(input_imgs, hue_delta)
    input_imgs = tf.image.random_brightness(input_imgs, brightness)
    input_imgs = tf.image.random_saturation(input_imgs, lsaturation,usaturation)       
    #merge process

    random_choice = tf.random_uniform([], 0.0, 1.0)
    
    input_imgs,label_imgs = projective_random_transform(input_imgs,label_imgs,projective_transform_angle,img_size)
    
    input_imgs,label_imgs = shift_img(input_imgs,label_imgs, width_shift_range, height_shift_range,img_size)
    
    input_imgs = tf.to_float(input_imgs) * scale 
    label_imgs = tf.to_float(label_imgs) * scale 
        
    if tf.less(random_choice, 0.5):
        input_imgs,label_imgs = flipran_img(input_imgs,label_imgs)
    else:
        input_imgs,label_imgs = rot_randomangle(input_imgs,label_imgs, angle)

    return input_imgs,label_imgs