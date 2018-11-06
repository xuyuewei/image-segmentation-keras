import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def smooth_l1(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    diff = tf.subtract(y_true_f, y_pred_f)
    abs_diff = tf.abs(diff)
    smooth = tf.map_fn(lambda x: tf.cond(x < 1, lambda x: (0.5*tf.square(x)), lambda x: (x-0.5)),abs_diff)
    loss = tf.reduce_mean(smooth)
    return loss
    
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def categorical_dice(y_true, y_pred):
    loss = losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
    
def regression_square_absolute(y_true, y_pred):
    loss = losses.mean_squared_error(y_true, y_pred) + losses.mean_absolute_error(y_true, y_pred)
    return loss

def cat_regression_loss(y_true, y_pred):
    loss = categorical_dice(y_true[0], y_pred[0]) + regression_square_absolute(y_true[1], y_pred[1]) + smooth_l1(y_true[1], y_pred[1])
    return loss
