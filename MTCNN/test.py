# import tensorflow as tf
# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c=tf.matmul(a,b)
# # Creates a session with log_device_placement set to True.
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#     print(sess.run(c))
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
img=cv2.imread("1.jpeg")
img_a=cv2.GaussianBlur(img,(501,501),0)
img_b=cv2.GaussianBlur(img,(501,501),500)
plt.figure("1")
plt.imshow(img)
plt.figure("2")
plt.imshow(img)
plt.show()
# img = cv2.resize(img, (img.shape[1]*2,img.shape[0]*2), interpolation=cv2.INTER_LINEAR)
# cv2.imshow("before",img)
# cv2.imshow("after",img_a)
# cv2.imshow("after2",img_b)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

