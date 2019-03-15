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

    def __init__(self, dataset_path,file_list, batch_size, num_dim, buffer_size=3000,shuffle=True):

        self.file_path=dataset_path
        self.file_list = file_list
        self.num_dim = num_dim

        # retrieve the data from the text file
        self._read_h5_file()

        # number of samples in the dataset
        self.data_size = len(self.img)

        # create dataset
        self.img = convert_to_tensor(self.img)
        self.labels = convert_to_tensor(np.array(self.labels))

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img, self.labels))

        # distinguish between train/infer. when calling the parsing functions

        # if mode == 'training':
        #     data = data.map(self._parse_function_train, num_threads=8,
        #               output_buffer_size=100*batch_size)
        #
        # elif mode == 'inference':
        #     data = data.map(self._parse_function_inference, num_threads=8,
        #               output_buffer_size=100*batch_size)

        # create a new dataset with batches of images
        if shuffle:
            data=data.shuffle(buffer_size=buffer_size)
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
    def _read_h5_file(self):
        """Read all the files in the dir"""

        self.img=[]
        self.labels = []
        # open dir and read pxx.txt to get information of samples
        for file in self.file_list:
            filepath=os.path.join(self.file_path,file)
            count=0;
            with h5py.File(filepath,'r') as f:
                for im in f['data'][0:128]:
                    im=np.array(im)
                    im=im.swapaxes(0,2)
                    im=np.flip(im)
                    im=np.rot90(im)
                    # get the normalized image by subtracting the AlextNet mean

                    new_image = im[:,:,::-1]
                    # new_image=im/256
                    # cv2.imshow("1",new_image)
                    # cv2.waitKey(0)
                    self.img.append(new_image)
                    if count%200== 0:
                        print("\r%s/3000" % str(count),end="")
                        sys.stdout.flush()
                    count+=1
                for label in f['label'][0:128]:
                    self.labels.append(label[0:2])
            print("\nData reading over...")
            break
