import os
import sys
import tensorflow as tf
import numpy as np
import h5py
import cv2

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
# used for normalizing the images

class ImageDataGenerator(object):

    def __init__(self,  file_path, batch_size, num_dim, buffer_size=3000, shuffle=True):

        self.num_dim = num_dim

        # retrieve the data from the text file
        # self.read_h5_file(file_path)

        # number of samples in the dataset
        # self.data_size=np.floor(3000/batch_size)*batch_size
        self.data_size=128
        # create dataset
        self.p_img = tf.placeholder(tf.float32, [self.data_size, 448, 448, 3])
        self.p_labels = tf.placeholder(tf.float32, [self.data_size,2])

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.p_img, self.p_labels))

        # create a new dataset with batches of images
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)
        data = data.batch(batch_size)

        self.data = data

    # read the unnormalized data(txt file)
    def _read_txt_file(self):
        """Read all the files in the dir"""
        dir_list=os.listdir(self.txt_file)
        dir_list.remove('readme.txt')
        self.img_paths = []
        self.eye = []
        self.gaze_direction=[]
        # open dir and read pxx.txt to get information of samples
        for d in dir_list:
            txt=os.path.join(self.txt_file,d,d+'.txt')
            with open(txt,'r') as f:
                lines = f.readlines()
                for line in lines:
                    items = line.split(' ')
                    self.img_paths.append(os.path.join(d,items[0]))
                    self.eye.append(items[-1])

    # read the normalized data(h5 file)
    def read_h5_file(self,file_path):
        """Read all the files in the dir"""
        # open dir and read pxx.txt to get information of samples
        self.img = []
        self.labels = []
        # for f_path in self.file_path:
        #     with h5py.File(f_path,'r') as f:
        #         for i in range(128):
        #             im=np.array(f['data'][i])
        #             im=im.swapaxes(0,2)
        #             im=np.flip(im,axis=1)
        #             im=np.rot90(im)
        #             new_image = im[:,:,::-1]
        #             # new_image=im/256
        #             # cv2.imshow("1",new_image)
        #             # cv2.waitKey(0)
        #             self.img.append(new_image)
        #             self.labels.append(f['label'][i][0:2])
        #             if i%100== 0:
        #                 print("\r%s/3000" % str(i),end="")
        #                 sys.stdout.flush()
        #     print("\nData reading over...")
        #     break
        count=0
        with h5py.File(file_path, 'r') as f:
            for im in f['data'][0:self.data_size]:
                im = np.array(im)
                im = im.swapaxes(0, 2)
                im = np.flip(im, axis=1)
                im = np.rot90(im)
                im = im[:, :, ::-1]
                # new_image=im/256
                # cv2.imshow("1",new_image)
                # cv2.waitKey(0)
                self.img.append(im)
                if count % 200 == 0:
                    print("\r%s/3000" % str(count), end="")
                    sys.stdout.flush()
                count += 1
            for label in f['label'][0:self.data_size]:
                self.labels.append(label[0:2])
        print("\nData reading over...")
