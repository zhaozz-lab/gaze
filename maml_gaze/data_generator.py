""" Code for loading data. """
import numpy as np
import os
import gc
import h5py
import random
import tensorflow as tf

from absl import flags
from tqdm import tqdm
import utils
# from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

class DataGenerator(object):

    def __init__(self, num_samples_per_class, batch_size, trian_list, test_list, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
            file_list: file waited to read
        """
        self.batch_size = batch_size
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = FLAGS.num_classes # by default 1 (only relevant for classification problems)
        self.norm_type='subtract_vgg'
        self.img_size = config.get('img_size', (36, 60, 3))
        self.dim_input=self.img_size
        # self.dim_input = np.prod(self.img_size)
        self.dim_output = 2
        # random.seed(1)
        # random.shuffle(trian_list)
        self.metatrain_character_folders = trian_list
        if FLAGS.test_set:
            self.metaval_character_folders = test_list
        else:
            self.metaval_character_folders = test_list

    def readmat(self,files,shuffle=True):
        # read .mat file
        gazes_l = np.vstack([files[idx]['eval_data']['gazes_L'] for idx in range(len(files))])
        gazes_r = np.vstack([files[idx]['eval_data']['gazes_R'] for idx in range(len(files))])
        headposes_l = np.vstack([files[idx]['eval_data']['headposes_L'] for idx in range(len(files))])
        headposes_r = np.vstack([files[idx]['eval_data']['headposes_R'] for idx in range(len(files))])
        images_l = np.vstack([files[idx]['eval_data']['imagesL'] for idx in range(len(files))])
        images_r = np.vstack([files[idx]['eval_data']['imagesR'] for idx in range(len(files))])

        tr_img = np.concatenate((images_l, images_r), axis=0)
        train_headposes = np.concatenate((headposes_l, headposes_r), axis=0)
        train_headposes = utils.pose2dir(train_headposes)
        train_gazes = np.concatenate((gazes_l, gazes_r), axis=0)
        num_instances = tr_img.shape[0]

        if shuffle:
            np.random.seed(1)
            shuffle_idx = np.arange(tr_img.shape[0])
            np.random.shuffle(shuffle_idx)
            tr_img = tr_img[shuffle_idx]
            train_headposes = train_headposes[shuffle_idx]
            train_gazes = train_gazes[shuffle_idx]

        train_images = np.zeros((num_instances, 36, 60, 3))
        for i in range(num_instances):
            train_images[i,:] = utils.get_normalized_image(tr_img[i,:], norm_type=self.norm_type)
        tr_img=None
        gc.collect()

        return train_images,train_headposes,train_gazes

    def make_data_tensor(self, train=True):
        if train:
            folders = self.metatrain_character_folders
            # number of tasks, not number of meta-iterations. (divide by metabatch size to measure)
            self.num_total_batches =num_total_batches= 240000
        else:
            folders = self.metaval_character_folders
            self.num_total_batches=num_total_batches = 512
        if type(folders) != list:
            folders=[folders]
        print("Reading .mat file")
        files = [h5py.File(folder) for folder in folders]
        img_data,headpose_data,gaze_data=[],[],[]
        for _ in files:
            img,headpose,gaze=self.readmat([_])
            img_data.append(img)
            headpose_data.append(headpose)
            gaze_data.append(gaze)
        # make list of files
        print('Generating filenames')
        img_data=np.array(img_data)
        headpose_data=np.array(headpose_data)
        gaze_data=np.array(gaze_data)

        if train:
            self.train_img_data=img_data_tensor=tf.placeholder(tf.float32,shape=(None,None,36,60,3))
            self.train_headpose_data=headpose_data_tensor=tf.placeholder(tf.float32,shape=(None,None,2))
            self.train_gaze_data=gaze_data_tensor=tf.placeholder(tf.float32,shape=(None,None,2))
        else:
            self.eval_img_data =img_data_tensor= tf.placeholder(tf.float32,shape=(None,None,36,60,3))
            self.eval_headpose_data=headpose_data_tensor=tf.placeholder(tf.float32,shape=(None,None,2))
            self.eval_gaze_data=gaze_data_tensor=tf.placeholder(tf.float32,shape=(None,None,2))
        folders=range(len(folders))
        samples=range(6000)
        all_filenames = []
        for _ in tqdm(range(num_total_batches)):
            sampled_character_folders = random.sample(folders, self.num_classes)
            random.shuffle(sampled_character_folders)
            sampler=lambda x:random.sample(x,self.num_samples_per_class)
            filenames=[[i,j] for i in sampled_character_folders \
                         for j in sampler(samples)]
            all_filenames.extend(filenames)
        all_filenames = np.vstack(all_filenames)
        examples_per_batch = self.num_classes * self.num_samples_per_class
        batch_image_size = self.batch_size * examples_per_batch

        dataset = tf.data.Dataset.from_tensor_slices((all_filenames)).repeat().\
                                  batch(batch_image_size)
        iter = dataset.make_initializable_iterator()
        if train:
            self.train_iterator_op = iter.initializer
        else:
            self.eval_iterator_op = iter.initializer

        filenames = iter.get_next()
        images = tf.gather_nd(img_data_tensor, filenames)
        headposes = tf.gather_nd(headpose_data_tensor, filenames)
        gazes = tf.gather_nd(gaze_data_tensor, filenames)
        all_image_batches, all_headpose_batches, all_gaze_batches = [], [],[]
        print('Manipulating image data to be right shape')
        # change the shape to meta_batch_size * way * shot
        for i in tqdm(range(self.batch_size)):
            image_batch = images[i*examples_per_batch:(i+1)*examples_per_batch]
            headpose_batch=headposes[i*examples_per_batch:(i+1)*examples_per_batch]
            gaze_batch = gazes[i*examples_per_batch:(i+1)*examples_per_batch]

            # arrange them by the order class first samples second meta-size last
            new_list, new_headpose_list,new_gaze_list = [], [], []
            for k in range(self.num_samples_per_class):
                class_idxs = tf.range(0, self.num_classes)
                class_idxs = tf.random_shuffle(class_idxs)

                true_idxs = class_idxs*self.num_samples_per_class + k
                new_list.append(tf.gather(image_batch,true_idxs))
                new_headpose_list.append(tf.gather(headpose_batch,true_idxs))
                new_gaze_list.append(tf.gather(gaze_batch, true_idxs))

            new_list = tf.concat(new_list, 0)  # has shape [self.num_classes*self.num_samples_per_class, self.dim_input]
            new_headpose_list = tf.concat(new_headpose_list, 0)
            new_gaze_list = tf.concat(new_gaze_list, 0)
            all_image_batches.append(new_list)
            all_headpose_batches.append(new_headpose_list)
            all_gaze_batches.append(new_gaze_list)

        all_image_batches = tf.stack(all_image_batches)
        all_headpose_batches = tf.stack(all_headpose_batches)
        all_gaze_batches = tf.stack(all_gaze_batches)
        return img_data,headpose_data,gaze_data, all_image_batches, all_headpose_batches, all_gaze_batches

