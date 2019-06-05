import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf

from swcnn import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime

# Path to the textfiles for the trainings and validation set


# Learning params
# learning_rate = 0.01
learning_rate = 0.001
num_epochs = 10
batch_size = 1

# Network params
dropout_rate = 0.5
num_classes = 2
train_layers = ['fc8', 'fc7', 'fc6']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./tensorboard"
checkpoint_path = "./checkpoints"
test_file = './test'
IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    d=os.listdir(test_file)
    filename=os.path.join(test_file,d[0])
    img_string = tf.read_file(filename)
    img_decoded = tf.image.decode_png(img_string, channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_centered = tf.subtract(img_resized, IMAGENET_MEAN)

    # RGB -> BGR
    img_bgr = img_centered[:, :, ::-1]
    img=tf.expand_dims(img_bgr,0)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()
cls=tf.argmax(score, 1)
# Start Tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    print("starting...............")

    ckpt=tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        img1,img2=sess.run([img_resized,img_centered])
        img=sess.run(img)
        score,cls=sess.run([score,cls],feed_dict={x:img,keep_prob:dropout_rate})
        print(score,cls)

