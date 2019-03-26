import os
import sys
import h5py
import numpy as np

import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

class ImageDataGenerator(object):
    def __init__(self,num_sample,batch_size,num_steps_epoch,face,gaze,norm_type):
        self.sample_idx=np.arange(num_sample)
        self.batch_size = batch_size
        self.num_steps_epoch = num_steps_epoch
        self.num_sample = num_sample

        self.face = face
        self.gaze = gaze

        self.norm_type = norm_type

    def get_data(self):
        while True:
            tr_face = np.zeros((self.batch_size, 448, 448, 3))
            tr_gaze = np.zeros((self.batch_size, 2))
            for i in range(self.batch_size):
                rand_index = np.random.randint(0, self.num_sample)
                index = self.sample_idx[rand_index]
                tr_face[i, :] = get_normalized_image(self.face[index, :], norm_type=self.norm_type)
                tr_gaze[i, :] = self.gaze[index, :]
            yield tr_face, tr_gaze

def get_normalized_image(image, norm_type):
    image=image.astype(np.float)
    if norm_type == 'subtract_alextnet':
        image[:, :, 0] = image[:, :, 0] - 123.68
        image[:, :, 1] = image[:, :, 1] - 116.779
        image[:, :, 2] = image[:, :, 2] - 103.939
    elif norm_type == '-1to1':
        image[:, :, 0] = image[:, :, 0] / 127.5 - 1.0
        image[:, :, 1] = image[:, :, 1] / 127.5 - 1.0
        image[:, :, 2] = image[:, :, 2] / 127.5 - 1.0
    elif norm_type == '0to1':
        image[:, :, 0] = image[:, :, 0] / 255.0
        image[:, :, 1] = image[:, :, 1] / 255.0
        image[:, :, 2] = image[:, :, 2] / 255.0
    else:
        raise ValueError("wrong norm_type!:"+norm_type)

    return image

def get_train_test_data(files,batch_size,do_shuffle=False):
    face=np.vstack(files[idx]['Data']['data'] for idx in range(len(files)))
    gaze=np.vstack(files[idx]['Data']['label'][:,0:2] for idx in range(len(files)))

    face = face.swapaxes(1, 3)
    face = np.rot90(face,k=3,axes=(1,2))
    num_instances = face.shape[0]
    num_steps_epoch = int(num_instances) // batch_size
    if do_shuffle:
        shuffle_idx = np.arange(num_instances)
        np.random.shuffle(shuffle_idx)
        face = face[shuffle_idx]
        gaze = gaze[shuffle_idx]

    print ("%s images loaded over" % (num_instances))
    print ("shape of 'images' is %s" % (face.shape,))
    print('Num steps epoch', num_steps_epoch)

    return face, gaze,num_instances,num_steps_epoch

def convert_to_unit_vector(angles):
    x = -tf.cos(angles[:, 0]) * tf.sin(angles[:, 1])
    y = -tf.sin(angles[:, 0])
    z = -tf.cos(angles[:, 1]) * tf.cos(angles[:, 1])
    norm = tf.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z

def compute_angle_error(labels, gaze):
    gaze_x, gaze_y, gaze_z = convert_to_unit_vector(gaze)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = gaze_x * label_x + gaze_y * label_y + gaze_z * label_z
    return tf.acos(angles) * 180 / np.pi