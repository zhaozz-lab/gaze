import tensorflow as tf
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('/home/leo/Desktop/Program/pycharm/gaze/MTCNN_DATA/data/MTCNN_model/PNet_landmark/')
    print(ckpt)