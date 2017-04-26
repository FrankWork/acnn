import tensorflow as tf
import numpy as np

r = tf.reshape(tf.range(12, dtype=tf.float32), [2,3,2])
u = tf.reshape(tf.range(4, dtype=tf.float32), [2,2])
g = tf.matmul(tf.reshape(r, [6, 2]), u)
g = tf.reshape(g, [2,3,2])

a0 = tf.matmul(r[0], u)
a1 = tf.matmul(r[1], u)
with tf.Session() as session:
  session.run(tf.global_variables_initializer())

  r, u, g,a0,a1 = session.run([r, u, g, a0, a1])
  print(r)
  print('*' * 10)
  print(u)
  print('*' * 10)
  print(g)
  print('*' * 10)
  print(a0)
  print('*' * 10)
  print(a1)
  print('*' * 10)
  



