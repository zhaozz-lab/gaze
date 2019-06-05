#!/usr/bin/python

from __future__ import print_function, division, absolute_import

import argparse
import os, os.path
import numpy as np
import h5py
import math
import gc
import glob
from sklearn.model_selection import KFold

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Model, load_model
from keras import backend as K

from train_tools_singleeye import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("fc1_size", type=int)
parser.add_argument("fc2_size", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("model_type", choices=['VGG16', 'VGG19'])
parser.add_argument("epoch", choices=['01', '02', '03', '04'])
parser.add_argument("gpu_num", choices=['0', '1', '2', '3'])

args = parser.parse_args()

model_type = args.model_type
fc1_size = args.fc1_size
fc2_size = args.fc2_size
batch_size = args.batch_size
epoch = args.epoch

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = args.gpu_num
# config.gpu_options.per_process_gpu_memory_fraction = 0.7

evaluate_train = False

eval_path = '/home/leo/Desktop/Dataset/MPIIGaze/Evaluation Subset/'
model_path = '/home/leo/Desktop/Dataset/MPIIGaze/Evaluation Subset/'

subjects_test_threefold = [
                           ['p00', 'p01', 'p03', 'p04'],
                           ['p05', 'p06', 'p07', 'p08'],
                           ['p09', 'p10', 'p11', 'p12', 'p13','p14']
]
subjects_train_threefold = [
                            ['p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p12', 'p13','p14'],
                            ['p00', 'p01', 'p03', 'p04', 'p09', 'p10', 'p11', 'p12', 'p13','p14'],
                            ['p00', 'p01', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08']
]
model_prefixes = ['']
model_suffixes = ['eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(batch_size)+'_1_'+epoch]
assert len(model_prefixes) == len(model_suffixes)

all_models_exist = True
for subjects_train, subjects_test in zip(subjects_train_threefold, subjects_test_threefold):
    for prefix, suffix in zip(model_prefixes, model_suffixes):
        if not os.path.isfile(model_path+"3Fold"+prefix+''.join(subjects_test)+suffix+".h5"):
            print('File does not exist', model_path+"3Fold"+prefix+''.join(subjects_test)+suffix+".h5")
            all_models_exist = False
            break

if os.path.isfile(eval_path+'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(batch_size)+'_'+epoch+'.txt'):
    model_evaluated = True
else:
    model_evaluated = False

assert all_models_exist == True and model_evaluated == False

scores_list = []
scores_list_models = []
for model_num in range(0, len(model_suffixes)):
    scores_list_models.append([])

for subjects_train, subjects_test in zip(subjects_train_threefold, subjects_test_threefold):
    print('subjects_test:', subjects_test)
    K.clear_session()
    set_session(tf.Session(config=config))

    models = []
    for prefix, suffix in zip(model_prefixes, model_suffixes):
        models.append(load_model(model_path+"3Fold"+prefix+''.join(subjects_test)+suffix+".h5", custom_objects={'accuracy_angle': accuracy_angle, 'angle_loss': angle_loss}))

    test_file_names = [eval_path + 'MPIIGaze_eval_' + subject + '.mat' for subject in subjects_test]
    test_files = [h5py.File(test_file_name) for test_file_name in test_file_names]

    test_images_l, test_images_r, test_gazes_l, test_gazes_r, \
    test_headposes_l, test_headposes_r, test_num = get_train_test_data_twoeyes(test_files, 'eval_data',
                                                                                  do_shuffle=True)

    test_images = np.concatenate((test_images_l, test_images_r), axis=0)
    tset_headposes = np.concatenate((test_headposes_l, test_headposes_r), axis=0)
    test_gazes = np.concatenate((test_gazes_l, test_gazes_r), axis=0)
    test_headposes = pose2dir(test_headposes)

    # testR1=test_images_R_1[0]
    # testR=np.reshape(np.array(testR1),[3,60,36])
    # testR=testR.swapaxes(0, 2)
    # plt.figure()
    # plt.imshow(testR)
    # plt.show()
    testimg= get_test_data_twoeyes(test_images)

    est_gazes = []
    for model in models:
        est_gazes.append(model.predict({'img_input': testimg, 'headpose_input':test_headposes}, verbose=1))

    total_errors = [0.0] * len(models)
    total_errorc = 0.0
    for i in range(test_num):
        errors = [0.0] * len(models)
        combined_gaze = [0.0, 0.0]
        for model_num in range(len(models)):
            errors[model_num] = accuracy_angle_openface(est_gazes[model_num][i], test_gazes[i])
            total_errors[model_num] += errors[model_num]
            combined_gaze[0] += est_gazes[model_num][i][0]
            combined_gaze[1] += est_gazes[model_num][i][1]
        combined_gaze[0] = combined_gaze[0] / len(models)
        combined_gaze[1] = combined_gaze[1] / len(models)
        errorc = accuracy_angle_openface(combined_gaze, test_gazes[i])
        total_errorc += errorc

    for model_num in range(len(models)):
        total_errors[model_num] = total_errors[model_num] / test_num
    total_errorc = total_errorc / test_num
    print('\n')

    for model_num in range(len(models)):
        print('Error model', model_num, ':', total_errors[model_num])
        scores_list_models[model_num].append(total_errors[model_num])

    print('Error combined', total_errorc)
    scores_list.append(total_errorc)

    for test_file in test_files:
        test_file.close()
    testimg_L, testimg_R = None, None 
    test_images_l, test_images_r, test_gazes_l, test_gazes_r,test_headposes_l,test_headposes_r = None, None, None, None
    test_headposes, test_gazes, test_images= None, None, None, None

    model = None

    gc.collect()
    K.clear_session()

with open(eval_path+'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(batch_size)+'_'+epoch+'.txt', "w") as f:
    print(scores_list)

    print('Ensemble: ' + str(np.mean(scores_list)) + ' +- ' + str(np.std(scores_list)))
    f.write('Ensemble: ' + str(np.mean(scores_list)) + ' +- ' + str(np.std(scores_list)) + '\n')
    for idx, score_model in enumerate(scores_list_models):
        print('Model ' + str(idx) + ': ' + str(np.mean(score_model)) + ' +- ' + str(np.std(score_model)))
        f.write('Model ' + str(idx) + ': ' + str(np.mean(score_model)) + ' +- ' + str(np.std(score_model)) + '\n')

