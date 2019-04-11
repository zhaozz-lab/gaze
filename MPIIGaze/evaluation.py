import os
import numpy as np
import cv2
import gc
import argparse
from datetime import datetime
from datetime import datetime
from matplotlib import pyplot as plt
from multiprocessing import Pool

import tensorflow as tf
from swcnn import SWCNN
from processing import *

def evaluate(val_lists,batch_size,train_layer,dataset_path,weight_path,filewriter_path,checkpoint_name):
    val_file = [h5py.File(val) for val in val_lists]
    val_face, val_gaze, val_num,num_step_epoch =get_train_test_data(val_file,batch_size,do_shuffle=False)
    val_generator=ImageDataGenerator(val_num,batch_size,num_step_epoch,val_face,val_gaze,norm_type='subtract_alextnet')

    val_dataset = tf.data.Dataset.from_generator(val_generator.get_data, (tf.float32,tf.float32),  (tf.TensorShape([batch_size,448,448,3]), tf.TensorShape([batch_size,2])))
    val_iterator = val_dataset.make_initializable_iterator()
    next_batch=val_iterator.get_next()

    input = tf.placeholder(tf.float32, [batch_size, 448, 448, 3])
    labels = tf.placeholder(tf.float32, [batch_size, 2])
    keep_prob = tf.placeholder(tf.float32)
    phase_train_test = tf.placeholder(tf.bool, name='phase_train_test')

    model = SWCNN(input, keep_prob, 2, train_layer, phase_train_test, weight_path)

    gaze = model.fc8
    with tf.name_scope("angle_error"):
        angle_error=tf.reduce_mean(compute_angle_error(labels,gaze))
    tf.summary.scalar('angle_error', angle_error)

    merged_summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter(filewriter_path)

    # ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(val_iterator.initializer)
        saver.restore(sess,checkpoint_name)
        writer.add_graph(sess.graph)
        # ----------------validation-----------------------

        print("{} Start validation".format(datetime.now()))
        averagemeter = 0
        for _ in range(val_generator.num_steps_epoch):
            face_batch, gaze_batch = sess.run(next_batch)
            result = sess.run(angle_error, feed_dict={input: face_batch,
                                                        labels: gaze_batch,
                                                        keep_prob: 1.,
                                                        phase_train_test: False})

            averagemeter+=result
        averagemeter=averagemeter/(val_generator.num_steps_epoch)
        print("{} Validation angle error = {:.5f}".format(datetime.now(),averagemeter))
    for file_ in val_file:
        file_.close()
    val_face, val_gaze = None, None
    model = None
    gc.collect()
    return averagemeter

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("learning_rate_base",type=float)
    parser.add_argument("epochs",type=int)
    parser.add_argument("batch_size",type=int)
    args = parser.parse_args()

    learning_rate_base = args.learning_rate_base
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate_decay=0.99
    dropout_rate = 1
    display_step = 1
    train_layer=['fc6','fc7','fc8']
    # dataset_path = "/home/leo/Desktop/Dataset/MPIIFaceGaze_normalized"
    dataset_path="E:/MPIIGaze/MPIIFaceGaze_normalized"
    weight_path = "./bvlc_alexnet.npy"
    filewriter_path = "./tensorboard"
    checkpoint_path = "./checkpoints"

    score_list=[]
    files_list = []
    for _ in os.listdir(dataset_path):
        if ".mat" in _ :
            files_list.append(os.path.join(dataset_path,_))
    # 4 fold cross validation
    val_list=[[files_list[0],files_list[1],files_list[13],files_list[14]],
                 [files_list[2],files_list[3],files_list[6],files_list[7]],
                 [files_list[4],files_list[5],files_list[9],files_list[11]]]
    for i in range(3):
        val_lists=val_list[i]
        print('loading data in {}'.format(val_lists))
        checkpoint_name = os.path.join(checkpoint_path, 'model_train_' + str(i) + '_' + str(num_epochs) + '_' + str(batch_size) + '.ckpt')
        with Pool(1) as pool:
            result=pool.apply(func=evaluate,args=(val_lists,
                                            batch_size,
                                            train_layer,
                                            dataset_path,
                                            weight_path,
                                            filewriter_path,
                                            checkpoint_name))
            score_list.append(result)
        '''
        # leave one out cross validation
        for i in range(15):
    
            val_list=files_list[i]
            print('loading data in {}'.format(val_list))
            val_file = [h5py.File(val_list)]
        '''
    t = datetime.now()
    txt = os.path.join("./evaluation_record",
                       str(learning_rate_base) + '_' + str(num_epochs) + "_" + str(batch_size) + '_' +
                       str(t.month) + '_' + str(t.day) + '_' + str(t.hour) + '_' + str(t.minute) + '_' + str(
                           t.second) + '.txt')
    with open(txt,'w') as f:
        print(score_list)
        print('Ensemble: ' + str(np.mean(score_list)) + ' +- ' + str(np.std(score_list)))
        f.write(checkpoint_name+'\n')
        f.write('The average angle is: ' + str(np.mean(score_list)) + ' +- ' + str(np.std(score_list)) + '\n')
