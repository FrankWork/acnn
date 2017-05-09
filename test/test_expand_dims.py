import tensorflow as tf

b = 2
d = 4
n = 3

x = tf.reshape(tf.range(b*n*d, dtype=tf.float32), [b, n, d])
y1 = tf.reshape(x, [b, n, d, 1])
y2 = tf.expand_dims(x, -1)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  x, y1, y2 = session.run([x, y1, y2])
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(y1)
  print('*' * 10)
  print(y2)