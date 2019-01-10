# coding:utf-8
import os
import random
import sys
import time
import cv2
import tensorflow as tf

from tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple
def _int64_feature(value):
    """Wrapper for insert int64 feature into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for insert bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def run(dataset_dir, net, output_dir, name='MTCNN', shuffling=False):
    """Runs the conversion operation.
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    # tfrecord name
    tf_filename = '%s/train_PNet_landmark.tfrecord' % (output_dir)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return
    # GET Dataset, and shuffling.
    dataset = get_dataset(dataset_dir, net=net)
    # filenames = dataset['filename']
    if shuffling:
        tf_filename = tf_filename + '_shuffle'
        # random.seed(12345454)
        random.shuffle(dataset)
    # Process dataset files.
    # write the data to tfrecord
    print('start writing')
    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if(i+1)%100==0:
                sys.stdout.write("\r>>%d/%dimages has been converted" %(i+1,len(dataset)))
            sys.stdout.flush()
            filename = image_example['filename']
            image=cv2.imread(filename)
            image_buffer = image.tostring()
            assert len(image.shape) == 3
            height = image.shape[0]
            width = image.shape[1]
            assert image.shape[2] == 3
            class_label=image_example['label']
            bbox=image_example['bbox']
            roi=[bbox['xmin'],bbox['ymin'],bbox['xmax'],bbox['ymax']]
            landmark = [bbox['xlefteye'], bbox['ylefteye'], bbox['xrighteye'], bbox['yrighteye'], bbox['xnose'],
                        bbox['ynose'],
                        bbox['xleftmouth'], bbox['yleftmouth'], bbox['xrightmouth'], bbox['yrightmouth']]
            example=tf.train.Example(features=tf.train.Features(feature={
                'image/encoded': _bytes_feature(image_buffer),
                'image/label': _int64_feature(class_label),
                'image/roi': _float_feature(roi),
                'image/landmark': _float_feature(landmark)
            }))


            tfrecord_writer.write(example.SerializeToString())
    # Finally, write the labels file:
    # labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
    print('\nFinished converting the MTCNN dataset!')

def get_dataset(dir, net='PNet'):
    # get file name , label and anotation
    # item = 'imglists/PNet/train_%s_raw.txt' % net
    dataset_dir = '%s/imglists/PNet/train_%s_landmark.txt' % (dir,net)
    with open(dataset_dir, 'r') as f:
        data=f.readlines()
    dataset = []
    for line in data:
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        # print(data_example['filename'])
        data_example['label'] = int(info[1])
        bbox=dict.fromkeys(['xmin','ymin','xmax','ymax','xlefteye','ylefteye','xrighteye','yrighteye',
                            'xnose','ynose','xleftmouth','yleftmouth','xrightmouth','yrightmouth'],0)
        if len(info) == 6:
            for index,key in enumerate(bbox):
                if index>3:
                    break
                bbox[key]=float(info[index+2])
        if len(info) == 12:
            for index,key in enumerate(bbox):
                if index<=3:
                    continue
                bbox[key]=float(info[index-2])
        data_example['bbox'] = bbox

        dataset.append(data_example)
    return dataset


if __name__ == '__main__':
    dir = '../DATA/'
    net = 'PNet'
    output_directory = '../DATA/imglists/PNet'
    run(dir, net, output_directory, shuffling=True)
