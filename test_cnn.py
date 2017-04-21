 import tensorflow as tf
import numpy as np


x = tf.ones([3, 784])
y_ = tf.ones([3, 10])


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print(h_conv1.get_shape())# (3, 28, 28, 32)
h_pool1 = max_pool_2x2(h_conv1)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  hp = session.run(h_pool1)
  print(hp.shape)#(3, 14, 14, 32)
# Given an input tensor of shape [batch, in_height, in_width, in_channels] and a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels], this op performs the following:

#     Flattens the filter to a 2-D matrix with shape [filter_height * filter_width * in_channels, output_channels].
#     Extracts image patches from the input tensor to form a virtual tensor of shape [batch, out_height, out_width, filter_height * filter_width * in_channels].
#     For each patch, right-multiplies the filter matrix and the image patch vector.
