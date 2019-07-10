""" Utility functions. """
import numpy as np
import cv2 as cv
import os
import math
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from absl import flags
# from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

def get_normalized_image(raw_image, norm_type):
    reshaped_image = raw_image.copy().reshape(36, 60, 3, order='F').astype(np.float,copy=False)

    if norm_type == 'subtract_vgg':
        reshaped_image[:, :, 0] = reshaped_image[:, :, 0] - 103.939
        reshaped_image[:, :, 1] = reshaped_image[:, :, 1] - 116.779
        reshaped_image[:, :, 2] = reshaped_image[:, :, 2] - 123.68
    elif norm_type == '-1to1':
        reshaped_image[:, :, 0] = reshaped_image[:, :, 0] / 127.5 - 1.0
        reshaped_image[:, :, 1] = reshaped_image[:, :, 1] / 127.5 - 1.0
        reshaped_image[:, :, 2] = reshaped_image[:, :, 2] / 127.5 - 1.0
    elif norm_type == '0to1':
        reshaped_image[:, :, 0] = reshaped_image[:, :, 0] / 255.0
        reshaped_image[:, :, 1] = reshaped_image[:, :, 1] / 255.0
        reshaped_image[:, :, 2] = reshaped_image[:, :, 2] / 255.0
    else:
        raise ValueError("wrong norm_type!:"+norm_type)

    return reshaped_image
def angle_loss(label,pred):
    # return tf.reduce_mean(tf.square(label-pred),axis=-1)
    return tf.reduce_sum(tf.square(label-pred))

def pose2dir(headpose):
    headpose = [cv.Rodrigues(p)[0] for p in headpose]
    headpose=np.asarray(headpose)
    pose=headpose[:,:,2]
    dir=np.zeros((headpose.shape[0],2))
    dir[:,0]=np.arcsin(pose[:,1])
    dir[:,1]=np.arctan2(pose[:,0],pose[:,2])
    return dir

def accuracy_angle(y_true, y_pred):
    from keras import backend as K
    import tensorflow as tf

    pred_x = -1*K.cos(y_pred[:,0])*K.sin(y_pred[:,1])
    pred_y = -1*K.sin(y_pred[:,0])
    pred_z = -1*K.cos(y_pred[:,0])*K.cos(y_pred[:,1])
    pred_norm = K.sqrt(pred_x*pred_x + pred_y*pred_y + pred_z*pred_z)

    true_x = -1*K.cos(y_true[:,0])*K.sin(y_true[:,1])
    true_y = -1*K.sin(y_true[:,0])
    true_z = -1*K.cos(y_true[:,0])*K.cos(y_true[:,1])
    true_norm = K.sqrt(true_x*true_x + true_y*true_y + true_z*true_z)

    angle_value = (pred_x*true_x + pred_y*true_y + pred_z*true_z) / (true_norm*pred_norm)
    K.clip(angle_value, -0.9999999999, 0.999999999)
    accuracy=(tf.acos(angle_value)*180.0)/math.pi
    return K.mean(accuracy)

def accuracy_angle_2(y_true, y_pred):
    pred_x = -1 * math.cos(y_pred[0]) * math.sin(y_pred[1])
    pred_y = -1 * math.sin(y_pred[0])
    pred_z = -1 * math.cos(y_pred[0]) * math.cos(y_pred[1])
    pred_norm = math.sqrt(pred_x * pred_x + pred_y * pred_y + pred_z * pred_z)

    true_x = -1 * math.cos(y_true[0]) * math.sin(y_true[1])
    true_y = -1 * math.sin(y_true[0])
    true_z = -1 * math.cos(y_true[0]) * math.cos(y_true[1])
    true_norm = math.sqrt(true_x * true_x + true_y * true_y + true_z * true_z)

    angle_value = (pred_x * true_x + pred_y * true_y + pred_z * true_z) / (true_norm * pred_norm)
    np.clip(angle_value, -0.9999999999, 0.999999999)
    return math.degrees(math.acos(angle_value))


def accuracy_angle_openface(y_true, y_pred):
    pred_x = -1 * math.cos(y_pred[0]) * math.sin(y_pred[1])
    pred_y = -1 * math.sin(y_pred[0])
    pred_z = -1 * math.cos(y_pred[0]) * math.cos(y_pred[1])
    pred = np.array([pred_x, pred_y, pred_z])
    pred = pred / np.linalg.norm(pred)

    true_x = -1 * math.cos(y_true[0]) * math.sin(y_true[1])
    true_y = -1 * math.sin(y_true[0])
    true_z = -1 * math.cos(y_true[0]) * math.cos(y_true[1])
    gt = np.array([true_x, true_y, true_z])
    gt = gt / np.linalg.norm(gt)

    angle_err = np.rad2deg(np.arccos(np.dot(pred, gt)))
    return angle_err