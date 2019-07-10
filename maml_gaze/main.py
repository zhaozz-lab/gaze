"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=60000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    10-shot gaze:
        python main.py --metatrain_iterations=10000 --meta_batch_size=32 --update_batch_size=10 --update_lr=0.001 --num_updates=1 --logdir=logs/MPIIgaze/
"""
import csv
import os
import numpy as np
import pickle
import random
import tensorflow as tf

from matplotlib import pyplot as plt
from data_generator import DataGenerator
from mamlgaze import MAMLGAZE
from absl import flags
from absl import app
# from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_integer('num_classes', 1, 'number of classes used in classification or number of subjects.')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')
flags.DEFINE_string('MPII','../../../Dataset/MPIIGaze/Evaluation Subset/','path of MPIIGaze dataset')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_float('momentum',0.9,'The momentum of momentum optimizer')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_list('fc_filters', [256,128], 'number of Full connected layer neurons')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', './logs', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot

def train(model, saver, sess, exp_string, data_generator,train_np,eval_np, feed_dict,resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 10000
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        # train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise

    for itr in range(resume_itr, FLAGS. pretrain_iterations + FLAGS.metatrain_iterations):
        # feed_dict = {}
        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1],
                                  model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])
        result = sess.run(input_tensors,feed_dict=feed_dict)
        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-4])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-3])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': prelosses: ' + str(np.mean(prelosses)) + ', postlosses: ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0 :
            input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]

            result = sess.run(input_tensors,feed_dict=feed_dict)
            print('Validation results: preaccuracy: ' + str(result[0]) + ', postaccuracy: ' + str(result[1]))

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

def test(model, saver, sess, exp_string, data_generator, feed_dict, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    for _ in range(len(data_generator.num_total_batches)):
        temp={model.meta_lr: 0.0}
        feed_dict.update(temp)
        result = sess.run([model.metaval_total_accuracies2], feed_dict=feed_dict)

        metaval_accuracies.append(result)

    metaval_accuracies = np.squeeze(np.array(metaval_accuracies))
    means = np.mean(metaval_accuracies,axis=0)
    stds = np.std(metaval_accuracies,axis=0)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_gaze' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.txt'
    with open(out_filename, 'a') as f:
        for index,_ in enumerate(means):
            f.write("accuracy %d : %f\n" % (index,_))

def main(argv):
    file_list=[]
    for _ in os.listdir(FLAGS.MPII):
        if ".mat" in _:
            file_list.append(os.path.join(FLAGS.MPII,_))
    # first one test
    train_list=file_list.copy()
    test_list=file_list[0]
    train_list.remove(file_list[0])

    if FLAGS.train:
        test_num_updates = 5
    else:
        test_num_updates = 10

    if FLAGS.train == False:
        FLAGS.meta_batch_size = 1
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
    data_generator = DataGenerator(FLAGS.update_batch_size*2, FLAGS.meta_batch_size,train_list,test_list)

    dim_output = data_generator.dim_output
    dim_input = data_generator.dim_input
    num_classes=data_generator.num_classes
    if FLAGS.train: # only construct training model if needed
        random.seed(5)
        image_np,headpose_np,gaze_np,image_tensor, headpose_tensor, gaze_tensor = data_generator.make_data_tensor()
        img_inputa = tf.slice(image_tensor, [0,0,0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1,-1,-1])
        img_inputb = tf.slice(image_tensor, [0,num_classes*FLAGS.update_batch_size, 0,0,0], [-1,-1,-1,-1,-1])
        headposea = tf.slice(headpose_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        headposeb = tf.slice(headpose_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        gazea = tf.slice(gaze_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
        gazeb = tf.slice(gaze_tensor, [0,num_classes*FLAGS.update_batch_size, 0], [-1,-1,-1])
        # note that the dict and feed_dict is different
        train_np=[image_np,headpose_np,gaze_np]
        input_tensors={'img_inputa':img_inputa,'img_inputb':img_inputb,'headposea':headposea,'headposeb':headposeb,'gazea':gazea,'gazeb':gazeb}

    random.seed(6)
    image_np, headpose_np, gaze_np, image_tensor, headpose_tensor, gaze_tensor = data_generator.make_data_tensor(train=False)
    img_inputa = tf.slice(image_tensor, [0, 0, 0,0,0], [-1, num_classes * FLAGS.update_batch_size, -1,-1,-1])
    img_inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0,0,0], [-1, -1, -1,-1,-1])
    headposea = tf.slice(headpose_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
    headposeb = tf.slice(headpose_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
    gazea = tf.slice(gaze_tensor, [0,0,0], [-1,num_classes*FLAGS.update_batch_size, -1])
    gazeb = tf.slice(gaze_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])

    eval_np = [image_np, headpose_np,  gaze_np]
    metaval_input_tensors = {'img_inputa': img_inputa, 'img_inputb': img_inputb,
                             'headposea': headposea, 'headposeb': headposeb,'gazea': gazea, 'gazeb': gazeb}

    model = MAMLGAZE(dim_input, dim_output, test_num_updates=test_num_updates)

    if FLAGS.train:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')

    model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')

    model.summ_op = tf.summary.merge_all()

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_update_batch_size == -1:
        FLAGS.train_update_batch_size = FLAGS.update_batch_size
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    # exp_string = 'gaze_'+'.mbs_'+str(FLAGS.meta_batch_size) + '.ubs_' + str(FLAGS.train_update_batch_size)\
    #              + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)

    exp_string = 'gaze_' + '.mbs_' + str(16) + '.ubs_' + str(FLAGS.train_update_batch_size) \
                 + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(FLAGS.train_update_lr)
    if FLAGS.fc_filters:
        exp_string += 'fc1_' + str(FLAGS.fc_filters[0])+'fc2_'+str(FLAGS.fc_filters[1])
    if FLAGS.stop_grad:
        exp_string += 'stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += 'layernorm'
    elif FLAGS.norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0
    tf.global_variables_initializer().run()
    if FLAGS.train:
        iterator=[data_generator.train_iterator_op, data_generator.eval_iterator_op]
        feed_dict={data_generator.train_img_data: train_np[0],
                   data_generator.train_headpose_data: train_np[1],
                   data_generator.train_gaze_data: train_np[2],
                   data_generator.eval_img_data: eval_np[0],
                   data_generator.eval_headpose_data: eval_np[1],
                   data_generator.eval_gaze_data: eval_np[2]}
    else:
        iterator = [data_generator.eval_iterator_op]
        feed_dict={data_generator.eval_img_data: eval_np[0],
                   data_generator.eval_headpose_data: eval_np[1],
                   data_generator.eval_gaze_data: eval_np[2]}
    sess.run(iterator,feed_dict)


    # show image
    # for i in range(5):
    #     train_img,eval_img =sess.run([input_tensors['img_inputa'],metaval_input_tensors['img_inputa']])
    #     print(i)
    # test=train_img[0][0]
    # test=test/255.0
    # plt.figure('train')
    # plt.imshow(test)
    #
    # test2 = eval_img[0][0]
    # test2 = test2 / 255.0
    # plt.figure('eval')
    # plt.imshow(test2)
    # plt.show()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, train_np,eval_np,feed_dict,resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator,feed_dict, test_num_updates)

if __name__ == "__main__":
    app.run(main)
