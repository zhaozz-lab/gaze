import tensorflow as tf
import numpy as np
import os
# img = [0,1,2,3,4,5,6,7,8]
# lbl = [0,1,2,3,4,5,6,7,8]
# images = tf.convert_to_tensor(img)
# labels = tf.convert_to_tensor(lbl)
# input_queue = tf.train.slice_input_producer([images,labels])
# sliced_img = input_queue[0]
# sliced_lbl = input_queue[1]

# img_batch, lbl_batch = tf.train.batch([images,labels], batch_size=1)
# with tf.Session() as sess:
#     coord   = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     for i in range(10):
#         image_batch, label_batch = sess.run([img_batch, lbl_batch])
#         print(image_batch, label_batch)
#         print('batch  ',i)
#         print('over')
#
#     coord.request_stop()
#     coord.join(threads)
#
#
# def test(x,y):
#     return x*2,y

# dataset = tf.data.Dataset.from_tensor_slices((images,labels)).repeat().batch(4)
# iter = dataset.make_one_shot_iterator()
# el=iter.get_next()
# with tf.Session() as sess:
#     for i in range(10):
#         print(sess.run(el))


import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

num_epoch = tf.Variable(0, name='global_step', trainable=False)
y = []
z = []
N = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('training')
    # for num_epoch in tqdm(range(N)):
    #     # 阶梯型衰减
    num_epoch=1000
    learing_rate1 = tf.train.exponential_decay(
        learning_rate=0.02, global_step=num_epoch, decay_steps=20, decay_rate=0.98, staircase=True)
    # 标准指数型衰减
    # learing_rate2 = tf.train.exponential_decay(
    #     learning_rate=0.001, global_step=num_epoch, decay_steps=10, decay_rate=0.96, staircase=False)
    lr1 = sess.run([learing_rate1])
    # lr2 = sess.run([learing_rate2])
    # y.append(lr1)
    # z.append(lr2)
    print('%.7f '% lr1[0])

