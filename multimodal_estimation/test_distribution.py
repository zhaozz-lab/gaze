#!/usr/bin/python

import numpy as np
import h5py
import gc
import math
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
evaluate_train=False


def get_data(files, label):
    # include imagesL, imagesR, gazes_L, gazes_R, origin_headposes, headposes_L, headposes_R
    gazes_l = np.vstack([files[idx][label]['gazes_L'] for idx in range(len(files))])
    gazes_r = np.vstack([files[idx][label]['gazes_R'] for idx in range(len(files))])
    headposes_l = np.vstack([files[idx][label]['headposes_L'] for idx in range(len(files))])
    headposes_r = np.vstack([files[idx][label]['headposes_R'] for idx in range(len(files))])
    orginheadpose= np.vstack([files[idx][label]['origin_headposes'] for idx in range(len(files))])

    num_instances = gazes_l.shape[0]
    # dimension = images_l.shape[1]

    print ("%s %s images loaded" % (num_instances, label))

    return gazes_l,gazes_r,headposes_l,headposes_r,orginheadpose,num_instances

def show_img(left,right):
    lefteye=np.reshape(left,(36,60,3),'F')[:,:,0]
    righteye=np.reshape(right,(36,60,3),'F')[:,:,0]
    fig,axes=plt.subplots(nrows=1,ncols=2,sharex=True)
    axes[0].imshow(righteye)
    axes[0].set_title('right eye')
    axes[1].imshow(lefteye)
    axes[1].set_title('left eye')
    plt.show()
def hist(headpose):
    headpose=headpose*180.0/math.pi
    plt.hist2d(headpose[:,1],headpose[:,0],bins=30,range=[[-50,50],[-50,50]],cmap='jet')
    plt.xlabel('h')
    plt.grid(True)
    cbar=plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.show()

def pose2dir(headpose):
    headpose = [cv.Rodrigues(p)[0] for p in headpose]
    headpose=np.asarray(headpose)
    pose=headpose[:,:,2]
    dir=np.zeros((headpose.shape[0],2))
    dir[:,0]=np.arcsin(pose[:,1])
    dir[:,1]=np.arctan2(pose[:,0],pose[:,2])
    return dir
eval_path = '/home/leo/Desktop/Dataset/MPIIGaze/Evaluation Subset/'
prefixes='MPIIGaze_eval_'
file_lists = ['p00', 'p01', 'p03', 'p04','p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p12', 'p13','p14']
# file_lists = ['p00']
file_names=[eval_path+prefixes+file_list+'.mat' for file_list in file_lists]
files=[h5py.File(file_name) for file_name in file_names]

test_images_L_1, test_images_R_1, test_gazes_1, test_headposes_1, test_num_1 = get_data(files, 'eval_data')
test_gazes_l,test_gazes_r,test_headposes_l,test_headposes_r,test_orginheadpose,test_num_instances= get_data(files, 'eval_data')
# testimg_L, testimg_R = get_test_data_twoeyes(test_images_L_1, test_images_R_1)
# for i in range(10):
#     k=i*10
#     show_img(test_images_L_1[k],test_images_R_1[k])
dir=pose2dir(test_headposes_1)
hist(dir)

for file in files:
    file.close()
testimg_L, testimg_R = None, None
test_images_L_1, test_images_R_1, test_gazes_1, test_headposes_1 = None, None, None, None
gc.collect()
