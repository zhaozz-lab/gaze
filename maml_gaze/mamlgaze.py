from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf

try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)
import utils
import gc
from absl import flags
# from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

class MAMLGAZE:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr,())
        self.test_num_updates = test_num_updates
        self.fcsize = FLAGS.fc_filters
        self.network = self.vgg16
        self.construct_weights = self.construct_fc_weights  # convenient for computing second gradient
        self.loss_func = utils.angle_loss

    def construct_model(self, input_tensors=None, prefix='metatrain'):
        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.img_inputa = tf.placeholder(tf.float32)
            self.img_inputb = tf.placeholder(tf.float32)
            self.headposea = tf.placeholder(tf.float32)
            self.headposeb = tf.placeholder(tf.float32)
            self.gazea = tf.placeholder(tf.float32)
            self.gazeb = tf.placeholder(tf.float32)
        else:
            self.img_inputa = input_tensors['img_inputa']
            self.img_inputb = input_tensors['img_inputb']
            self.headposea = input_tensors['headposea']
            self.headposeb = input_tensors['headposeb']
            self.gazea = input_tensors['gazea']
            self.gazeb = input_tensors['gazeb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                untrain_weights,weights =  self.untrain_weights,self.weights
            else:
                # Define the weights
                self.untrain_weights,self.weights = untrain_weights,weights = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """

                img_inputa,img_inputb,headposea,headposeb,gazea,gazeb = inp
                task_outputbs, task_lossesb,task_accuraciesb = [], [], []

                task_outputa = self.network(img_inputa,headposea,untrain_weights,weights,reuse=reuse)
                task_lossa = self.loss_func(task_outputa, gazea)

                # test tf.Graph memory usage
                # with tf.Session() as sess:
                #     train_writer = tf.summary.FileWriter('./logs', sess.graph)
                #     print(1)

                grads = tf.gradients(task_lossa, list(weights.values()))
                # every task sample in meta learning, accumulation should be 0
                # accumulation = dict(zip(weights.keys(), [tf.Variable(0.0) for i in range(len(weights.keys()))]))

                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
                # for key in weights.keys():
                #     accumulation[key]=FLAGS.momentum * accumulation[key] + gradients[key]
                # fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*accumulation[key] for key in weights.keys()]))

                output = self.network(img_inputb,headposeb, untrain_weights,fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, gazeb))

                for j in range(num_updates - 1):
                    loss = self.loss_func(self.network(img_inputa,headposea,untrain_weights,fast_weights, reuse=True), gazea)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    # for key in fast_weights.keys():
                    #     accumulation[key] = FLAGS.momentum * accumulation[key] + gradients[key]
                    # fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*accumulation[key] for key in fast_weights.keys()]))

                    output = self.network(img_inputb,headposeb, untrain_weights,fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, gazeb))
                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]
                task_accuracya = utils.accuracy_angle(task_outputa,gazea)
                for j in range(num_updates):
                        task_accuraciesb.append(utils.accuracy_angle(task_outputbs[j],gazeb))
                task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.img_inputa[0], self.img_inputb[0], self.headposea[0], self.headposeb[0],self.gazea[0],self.gazeb[0]), False)

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates,tf.float32, [tf.float32]*num_updates]
            result = tf.map_fn(task_metalearn, elems=(self.img_inputa, self.img_inputb, self.headposea, self.headposeb,self.gazea,self.gazeb),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result

        #----------  Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        ## Summaries
        tf.summary.scalar(prefix+'Pre_update_loss', total_loss1)
        tf.summary.scalar(prefix+'Pre_update_accuracy', total_accuracy1)

        # for j in range(num_updates):
        #     tf.summary.scalar(prefix+'Post_update_loss_step_' + str(j+1), total_losses2[j])
        #     tf.summary.scalar(prefix+'Post_update_accuracy_step_' + str(j+1), total_accuracies2[j])
        tf.summary.scalar(prefix + 'Post_update_loss_step_' + str(num_updates), total_losses2[num_updates-1])
        tf.summary.scalar(prefix+'Post_update_accuracy_step_' + str(num_updates), total_accuracies2[num_updates-1])

    def construct_fc_weights(self):
        untrain_weights={}
        weights={}
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        vgg16_npy_path='./data/vgg16.npy'
        data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        conv_trainable=False
        select_trainable=True

        untrain_weights['conv1_1w'] = tf.get_variable(name='conv1_1w', trainable=conv_trainable, shape=data_dict['conv1_1'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv1_1'][0]))
        untrain_weights['conv1_1b'] = tf.get_variable(name='conv1_1b', trainable=conv_trainable, shape=data_dict['conv1_1'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv1_1'][1]))
        untrain_weights['conv1_2w'] = tf.get_variable(name='conv1_2w', trainable=conv_trainable,shape=data_dict['conv1_2'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv1_2'][0]))
        untrain_weights['conv1_2b'] = tf.get_variable(name='conv1_2b', trainable=conv_trainable,shape=data_dict['conv1_2'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv1_2'][1]))

        untrain_weights['conv2_1w'] = tf.get_variable(name='conv2_1w', trainable=conv_trainable,shape=data_dict['conv2_1'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv2_1'][0]))
        untrain_weights['conv2_1b'] = tf.get_variable(name='conv2_1b', trainable=conv_trainable, shape=data_dict['conv2_1'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv2_1'][1]))
        untrain_weights['conv2_2w'] = tf.get_variable(name='conv2_2w', trainable=conv_trainable,shape=data_dict['conv2_2'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv2_2'][0]))
        untrain_weights['conv2_2b'] = tf.get_variable(name='conv2_2b', trainable=conv_trainable,shape=data_dict['conv2_2'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv2_2'][1]))

        untrain_weights['conv3_1w']=tf.get_variable(name='conv3_1w', trainable=conv_trainable, shape=data_dict['conv3_1'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv3_1'][0]))
        untrain_weights['conv3_1b'] = tf.get_variable(name='conv3_1b', trainable=conv_trainable, shape=data_dict['conv3_1'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv3_1'][1]))
        untrain_weights['conv3_2w'] = tf.get_variable(name='conv3_2w', trainable=conv_trainable,shape=data_dict['conv3_2'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv3_2'][0]))
        untrain_weights['conv3_2b'] = tf.get_variable(name='conv3_2b', trainable=conv_trainable,shape=data_dict['conv3_2'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv3_2'][1]))
        untrain_weights['conv3_3w'] = tf.get_variable(name='conv3_3w', trainable=conv_trainable,shape=data_dict['conv3_3'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv3_3'][0]))
        untrain_weights['conv3_3b'] = tf.get_variable(name='conv3_3b', trainable=conv_trainable, shape=data_dict['conv3_3'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv3_3'][1]))

        untrain_weights['conv4_1w'] = tf.get_variable(name='conv4_1w', trainable=conv_trainable,shape=data_dict['conv4_1'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv4_1'][0]))
        untrain_weights['conv4_1b'] = tf.get_variable(name='conv4_1b', trainable=conv_trainable, shape=data_dict['conv4_1'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv4_1'][1]))
        untrain_weights['conv4_2w'] = tf.get_variable(name='conv4_2w', trainable=conv_trainable,shape=data_dict['conv4_2'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv4_2'][0]))
        untrain_weights['conv4_2b'] = tf.get_variable(name='conv4_2b', trainable=conv_trainable,shape=data_dict['conv4_2'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv4_2'][1]))
        untrain_weights['conv4_3w'] = tf.get_variable(name='conv4_3w', trainable=conv_trainable,shape=data_dict['conv4_3'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv4_3'][0]))
        untrain_weights['conv4_3b'] = tf.get_variable(name='conv4_3b', trainable=conv_trainable,shape=data_dict['conv4_3'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv4_3'][1]))

        untrain_weights['conv5_1w'] = tf.get_variable(name='conv5_1w', trainable=conv_trainable,shape=data_dict['conv5_1'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv5_1'][0]))
        untrain_weights['conv5_1b'] = tf.get_variable(name='conv5_1b', trainable=conv_trainable,shape=data_dict['conv5_1'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv5_1'][1]))
        untrain_weights['conv5_2w'] = tf.get_variable(name='conv5_2w', trainable=conv_trainable,shape=data_dict['conv5_2'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv5_2'][0]))
        untrain_weights['conv5_2b'] = tf.get_variable(name='conv5_2b', trainable=conv_trainable,shape=data_dict['conv5_2'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv5_2'][1]))
        weights['conv5_3w'] = tf.get_variable(name='conv5_3w', trainable=select_trainable,shape=data_dict['conv5_3'][0].shape,
                                              initializer=tf.constant_initializer(data_dict['conv5_3'][0]))
        weights['conv5_3b'] = tf.get_variable(name='conv5_3b', trainable=select_trainable, shape=data_dict['conv5_3'][1].shape,
                                              initializer=tf.constant_initializer(data_dict['conv5_3'][1]))

        weights['fc6w']=tf.get_variable('fc6_weights',[2*2*512,self.fcsize[0]],
                                        trainable=select_trainable,initializer=fc_initializer)
        weights['fc6b']=tf.Variable(tf.zeros(self.fcsize[0]),name='fc6_biases',trainable=select_trainable)
        weights['fc7w']=tf.get_variable('fc7_weights',[self.fcsize[0]+2,self.fcsize[1]],
                                        trainable=select_trainable,initializer=fc_initializer)
        weights['fc7b']=tf.Variable(tf.zeros(self.fcsize[1]),name='fc7_biases',trainable=select_trainable)
        weights['fc8w']=tf.get_variable('fc8_weights',[self.fcsize[1],2],
                                        trainable=select_trainable,initializer=fc_initializer)
        weights['fc8b']=tf.Variable(tf.zeros(2),name='fc8_biases',trainable=select_trainable)

        return untrain_weights,weights

    def vgg16(self,img_iput,headpose_input,untrain_weights,weights,reuse):

        def get_conv_filter(name):
            return tf.constant(data_dict[name][0], name="filter")

        def get_bias(name):
            return tf.constant(data_dict[name][1], name="biases")

        def get_fc_weight(name):
            return tf.constant(data_dict[name][0], name="weights")

        def avg_pool(bottom, name):
            return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

        def max_pool(bottom, name):
            return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

        def conv_layer(bottom, name, trainable=False):
            with tf.variable_scope(name,reuse=reuse):
                if trainable==False:
                    filt=untrain_weights[name+'w']
                    conv_biases = untrain_weights[name + 'b']
                else:
                    filt = weights[name + 'w']
                    conv_biases = weights[name + 'b']
                conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
                bias = tf.nn.bias_add(conv, conv_biases)
                relu = tf.nn.relu(bias)
                return relu

        def fc_layer2(bottom,name):
            with tf.variable_scope(name):
                # shape = bottom.get_shape().as_list()
                # dim = 1
                # for d in shape[1:]:
                #     dim *= d
                # x = tf.reshape(bottom, [-1, dim])

                w=weights[name+'w']
                b=weights[name+'b']
                # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
                fc = tf.nn.bias_add(tf.matmul(bottom, w), b)
                return fc

        conv1_1 = conv_layer(img_iput, "conv1_1")
        conv1_2 = conv_layer(conv1_1, "conv1_2")
        pool1 = max_pool(conv1_2, 'pool1')

        conv2_1 = conv_layer(pool1, "conv2_1")
        conv2_2 = conv_layer(conv2_1, "conv2_2")
        pool2 = max_pool(conv2_2, 'pool2')

        conv3_1 = conv_layer(pool2, "conv3_1")
        conv3_2 = conv_layer(conv3_1, "conv3_2")
        conv3_3 = conv_layer(conv3_2, "conv3_3")
        pool3 = max_pool(conv3_3, 'pool3')

        conv4_1 = conv_layer(pool3, "conv4_1")
        conv4_2 = conv_layer(conv4_1, "conv4_2")
        conv4_3 = conv_layer(conv4_2, "conv4_3")
        pool4 = max_pool(conv4_3, 'pool4')

        conv5_1 = conv_layer(pool4, "conv5_1")
        conv5_2 = conv_layer(conv5_1, "conv5_2")
        conv5_3 = conv_layer(conv5_2, "conv5_3",True)
        pool5 = max_pool(conv5_3, 'pool5')

        flatten=tf.layers.flatten(pool5)
        fc6 = fc_layer2(flatten, "fc6")
        # assert fc6.get_shape().as_list()[1:] == [self.fcsize[0]]
        fc6 = tf.concat([fc6, headpose_input],axis=-1)
        relu6=utils.normalize(fc6,activation=tf.nn.relu,reuse=reuse,scope='fc6')

        fc7 = fc_layer2(relu6, "fc7")
        relu7=utils.normalize(fc7,activation=tf.nn.relu,reuse=reuse,scope='fc7')

        output = fc_layer2(relu7, "fc8")

        data_dict = None
        gc.collect()

        return output

