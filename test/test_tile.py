import tensorflow as tf

b = 2
nr = 3
d = 4

x = tf.reshape(tf.range(b*d, dtype=tf.float32), [b, d])
y = tf.tile(tf.expand_dims(x, axis=1), [1, nr, 1])

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  x, y= session.run([x, y])
 
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(y.shape)
  print(y)
  