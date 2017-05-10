import tensorflow as tf

b = 2
n = 3
d = 4


x = tf.reshape(tf.range(b*n*d, dtype=tf.float32), [b, n, d])
y = tf.norm(x, axis=2)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  x , y = session.run([x, y])
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(y)