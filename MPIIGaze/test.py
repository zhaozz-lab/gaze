import tensorflow as tf
filewriter_path='./tensorboard'

t=tf.Variable([0,1,2])

writer = tf.summary.FileWriter(filewriter_path)
with tf.Session() as sess:
        pass
#     writer.add_graph(sess.graph)
tf.reset_default_graph()
with tf.Session() as sess:
    writer.add_graph(sess.graph)