import tensorflow as tf
a=tf.Variable(0,dtype=tf.float16)
ema=tf.train.ExponentialMovingAverage(0.5)
op=ema.apply([a])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        with tf.control_dependencies([op]):
            sess.run(tf.assign(a,i+1))
            print(sess.run(ema.average(a)))