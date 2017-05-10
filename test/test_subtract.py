import tensorflow as tf


x = tf.reshape(tf.range(12, dtype=tf.float32), [3, 4])
y = tf.ones([3, 4])
# z = tf.subtract(x, y)
z = x - y

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  x, y, z = sess.run([x, y, z])
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(y)
  print('*' * 10)
  print(z)