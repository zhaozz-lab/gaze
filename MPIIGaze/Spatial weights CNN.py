import os
import numpy as np
import cv2
import gc
import argparse
import threading
from datetime import datetime
from multiprocessing import Process

import tensorflow as tf
from swcnn import SWCNN
from processing import *
def train(train_file_list,
          num_epochs,
          learning_rate_base,
          batch_size,
          learning_rate_decay,
          dropout_rate,
          display_step,
          train_layer,
          dataset_path,
          weight_path,
          filewriter_path,
          checkpoint_path,
          i):
    train_file = [h5py.File(train_file_name) for train_file_name in train_file_list]
    train_face, train_gaze, train_num, num_step_epoch = get_train_test_data(train_file, batch_size, do_shuffle=True)
    generator = ImageDataGenerator(train_num, batch_size, num_step_epoch, train_face, train_gaze,
                                   norm_type='subtract_alextnet')

    dataset = tf.data.Dataset.from_generator(generator.get_data, (tf.float32, tf.float32), (
    tf.TensorShape([batch_size, 448, 448, 3]), tf.TensorShape([batch_size, 2])))
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    input = tf.placeholder(tf.float32, [batch_size, 448, 448, 3])
    labels = tf.placeholder(tf.float32, [batch_size, 2])
    keep_prob = tf.placeholder(tf.float32)
    model = SWCNN(input, keep_prob, 2, train_layer, weight_path)

    gaze = model.fc8

    with tf.name_scope("l1_loss"):
        loss=tf.losses.absolute_difference(labels, gaze,reduction='none')
        loss=tf.reduce_mean(loss)
        # loss=tf.reduce_mean(tf.squared_difference(labels,gaze))
    tf.summary.scalar('l1_loss', loss)

    global_step=tf.Variable(0,trainable=False)
    with tf.name_scope("learning_rate"):
        learning_rate=tf.train.exponential_decay(learning_rate_base,
                                                 global_step,
                                                 num_step_epoch,
                                                 learning_rate_decay)
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.9, beta2=0.95)
        train_op=optimizer.minimize(loss,global_step=global_step)

    with tf.name_scope("angle_error"):
        angle_error=tf.reduce_mean(compute_angle_error(labels,gaze))
    tf.summary.scalar('angle_error', angle_error)

    merged_summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter(filewriter_path)

    checkpoint_name = os.path.join(checkpoint_path,'model_train_' +str(i)+'_'+ str(num_epochs) + '_' + str(batch_size) + '.ckpt')
    model_exit=False
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt:
        model_exit=True
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        if model_exit:
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            model.load_initial_weights(sess)
        writer.add_graph(sess.graph)
        # ----------------train-----------------------
        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))
        for epoch in range(num_epochs):
            print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
            for step in range(generator.num_steps_epoch):

                face_batch, gaze_batch = sess.run(next_batch)

                angle_er,losses, _ = sess.run([angle_error,loss, train_op],feed_dict={input: face_batch,
                                                                                      labels: gaze_batch,
                                                                                      keep_prob: dropout_rate})
                print("step: {} losses: {}  angle error: {}".format(step,losses,angle_er))
                # if step % display_step == 0:
                #     s = sess.run(merged_summary, feed_dict={input: face_batch,
                #                                             labels: gaze_batch,
                #                                             keep_prob: 1.})
                #     writer.add_summary(s, epoch * generator.num_steps_epoch + step)
        # ----------------save---------------------------
        print("{} Saving checkpoint of model...".format(datetime.now()))
        # save checkpoint of the model
        save_path = saver.save(sess, checkpoint_name)
        print("{} Model checkpoint saved at "
              "{}".format(datetime.now(),checkpoint_name))

    for file in train_file:
        file.close()
    train_face, train_gaze = None, None
    tf.reset_default_graph()
    gc.collect()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("learning_rate_base",type=float)
    parser.add_argument("epochs",type=int)
    parser.add_argument("batch_size",type=int)
    args = parser.parse_args()

    learning_rate_base = args.learning_rate_base
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate_decay=0.95
    dropout_rate = 1
    display_step = 5
    train_layer=['fc6','fc7','fc8']
    dataset_path = "/home/leo/Desktop/Dataset/MPIIFaceGaze_normalized"
    # dataset_path="E:/MPIIGaze/MPIIFaceGaze_normalized"
    weight_path = "./bvlc_alexnet.npy"
    filewriter_path = "./tensorboard"
    checkpoint_path = "./checkpoints"
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.isdir(filewriter_path):
        os.mkdir(filewriter_path)
    files_list = []
    for _ in os.listdir(dataset_path):
        if ".mat" in _ :
            files_list.append(os.path.join(dataset_path,_))
    # leave one out cross validation
    for i in range(15):
        i=0
        train_list=files_list
        val_list=files_list[i]
        train_list.remove(val_list)
        train_lists=[train_list[0:4],train_list[4:8],train_list[8:12],train_list[12:14]]
        # train by three steps

        for index,train_file_list in enumerate(train_lists):
            print('loading data in {}'.format(train_file_list))
            p = Process(target=train, args=(train_file_list,
                                            num_epochs,
                                            learning_rate_base,
                                            batch_size,
                                            learning_rate_decay,
                                            dropout_rate,
                                            display_step,
                                            train_layer,
                                            dataset_path,
                                            weight_path,
                                            filewriter_path,
                                            checkpoint_path,
                                            i))
            p.start()
            p.join()
            print('Phase %d train over'%(index))

        gc.collect()
        break