import os
import numpy as np
import cv2

import tensorflow as tf
from swcnn import SWCNN
from datetime import datetime
from preprocessing import ImageDataGenerator

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

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
def train(sess,train_batches_per_epoch,next_batch,loss,train_op,input,labels,keep_prob,dropout_rate,writer,merged_summary,epoch):
    for step in range(train_batches_per_epoch):

        # get next batch of data
        img_batch, label_batch = sess.run(next_batch)

        # And run the training op
        losses, _ = sess.run([loss, train_op], feed_dict={input: img_batch,
                                                          labels: label_batch,
                                                          keep_prob: dropout_rate})
        print(losses)
        # Generate summary with the current batch of data and write to file
        if step % display_step == 0:
            s = sess.run(merged_summary, feed_dict={input: img_batch,
                                                    labels: label_batch,
                                                    keep_prob: 1.})
            writer.add_summary(s, epoch * train_batches_per_epoch + step)
def main(args):
    learning_rate = args[0]
    num_epochs = args[1]
    batch_size = args[2]
    # Network params
    dropout_rate = args[3]
    num_dim = args[4]
    # How often we want to write the tf.summary data to disk
    display_step = args[5]
    # path for dataset
    dataset_path = args[6]
    # Path for tf.summary.FileWriter and to store model checkpoints
    weight_path = args[7]
    filewriter_path = args[8]
    checkpoint_path = args[9]
    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.isdir(filewriter_path):
        os.mkdir(filewriter_path)
    # load data
    leave_one_out=args[10]
    train_list = []
    val_list=[]
    dir = os.listdir(dataset_path)
    for _ in dir:
        if ".h5" in _ :
            if str(leave_one_out) in _ :
                path=os.path.join(dataset_path,_)
                val_list.append(path)
            else:
                path = os.path.join(dataset_path, _)
                train_list.append(path)
    with tf.device("/cpu:0"):
        train_data=ImageDataGenerator(file_path='/',
                                      batch_size=batch_size,
                                      num_dim=num_dim,
                                      shuffle=True)
        # val_data = ImageDataGenerator(val_list,
        #                               batch_size=batch_size,
        #                               num_dim=num_dim,
        #                               shuffle=False)
        iterator = tf.data.Iterator.from_structure(train_data.data.output_types,
                                           train_data.data.output_shapes)
        next_batch = iterator.get_next()
    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(train_data.data)
    # validation_init_op=iterator.make_initializer(val_data.data)

    input = tf.placeholder(tf.float32, [batch_size, 448, 448, 3])
    input = tf.subtract(input, IMAGENET_MEAN)
    labels = tf.placeholder(tf.float32, [batch_size, num_dim])
    keep_prob=tf.placeholder(tf.float32)
    # the layers need to train
    train_layers = ['sconv1', 'sconv2', 'sconv3', 'fc8', 'fc7', 'fc6']
    # Initialize model
    model = SWCNN(input, keep_prob, num_dim, train_layers,weight_path)

    # Link variable to model output
    gaze = model.fc8
    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
    # op for calculating the loss
    with tf.name_scope("l1_loss"):
        loss=tf.losses.absolute_difference(labels, gaze,reduction='none')
        loss=tf.reduce_mean(loss)
        # Add the loss to summary
    tf.summary.scalar('l1_loss', loss)

    # Train op
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        # g1=tf.identity(gradients[0])
        # the gradient with respect to W show be normalised
        for i in range(6):
            gradients[i]=tf.divide(gradients[i],float(256))
        gradients = list(zip(gradients, var_list))
        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)
    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Evaluation op: Average angle error
    with tf.name_scope("angle_error"):
        angle_error=tf.reduce_mean(compute_angle_error(labels,gaze))

    # Add the accuracy to the summary
    tf.summary.scalar('angle_error', angle_error)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.floor(train_data.data_size/batch_size))
    # val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))

    # Start Tensorflow session
    with tf.Session() as sess:

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        # Load the pretrained weights into the non-trainable layer
        model.load_initial_weights(sess)
        # ----------------train-----------------------

        for train_path in train_list:
            train_data.read_h5_file(train_path)
            sess.run(training_init_op,feed_dict={train_data.p_img:np.array(train_data.img),
                                                 train_data.p_labels:np.array(train_data.labels)})

            print("{} Start training...".format(datetime.now()))
            print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                              filewriter_path))
            # Loop over number of epochs
            for epoch in range(num_epochs):
                sess.run(training_init_op,feed_dict={train_data.p_img:np.array(train_data.img),
                                                 train_data.p_labels:np.array(train_data.labels)})
                print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

                train(sess, train_batches_per_epoch, next_batch, loss, train_op, input, labels, keep_prob, dropout_rate,
                      writer, merged_summary,epoch)

        # ----------------validation-----------------------
        '''
        print("{} Start validation".format(datetime.now()))
        averagemeter = AverageMeter()
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run([val_data.img_batch,val_data.label_batch])
            result = sess.run(angle_error, feed_dict={input: img_batch,
                                                labels: label_batch,
                                                keep_prob: 1.})
            averagemeter.update(result,_)
        average_angle_error=averagemeter.avg
        print("{} Validation angle error = {:.5f}".format(datetime.now(),
                                                       average_angle_error))
        print("{} Saving checkpoint of model...".format(datetime.now()))

        coord.request_stop()
        coord.join(threads)

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path,
                                       'model_epoch'+str(epoch+1)+'.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                       checkpoint_name))
        '''
if __name__=='__main__':
    # learning parameter
    learning_rate = 0.00001
    num_epochs = 10
    batch_size = 64
    # Network params
    dropout_rate = 1
    num_dim = 2
    # How often we want to write the tf.summary data to disk
    display_step = 1
    # path for dataset
    dataset_path = "/home/leo/Desktop/Dataset/MPIIFaceGaze_normalized"
    # Path for tf.summary.FileWriter and to store model checkpoints
    weight_path = "./AlexNet/bvlc_alexnet.npy"
    filewriter_path = "./tensorboard"
    checkpoint_path = "./checkpoints"
    leave_one_out=14
    args=[learning_rate,
          num_epochs,
          batch_size,
          dropout_rate,
          num_dim,
          display_step,
          dataset_path,
          weight_path,
          filewriter_path,
          checkpoint_path,
          leave_one_out]
    main(args)