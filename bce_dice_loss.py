import numpy as np
from keras.losses import binary_crossentropy

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = np.reshape(y_true, -1)
    y_pred_f = np.reshape(y_pred, -1)
    intersection = np.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
