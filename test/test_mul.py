import tensorflow as tf

b = 1
n = 4
d = 3
k = 3

dc = 3

x = tf.reshape(tf.range(b*n*dc, dtype=tf.float32), [b, n, dc])
a = tf.reshape(tf.range(b*n, dtype=tf.float32), [b, n])
y = tf.multiply(x, tf.reshape(a, [b, n, 1]))

x = tf.transpose(x, perm=[0, 2, 1])
y = tf.transpose(y, perm=[0, 2, 1])

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  e, x, y = session.run([x, a, y])
  print('*' * 10)
  print(e)
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(y)
