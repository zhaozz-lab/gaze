import tensorflow as tf
import numpy as np
a=[1.0]
img=tf.convert_to_tensor(np.array(a))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(img))