import tensorflow as tf
import numpy as np


x = tf.ones([2, 3, 3, 1])
W = tf.Variable(tf.ones([2, 3, 1, 2]))
# b = tf.Variable(tf.constant(0.1, shape=[2]))

conv = tf.nn.conv2d(x, W, strides=[1, 1, 3, 1], padding='SAME')#  + b
print(conv.get_shape())# 

# relu = tf.nn.relu(conv)
# pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as session:
  session.run(tf.global_variables_initializer())

  x, w, conv = session.run([x, W, conv])
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(w)
  print('*' * 10)
  print(conv)
