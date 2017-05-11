import tensorflow as tf

b = 2
n = 3
d = 4


x = tf.reshape(tf.range(b*n*d, dtype=tf.float32), [b, n, d])
e = tf.reshape(tf.range(b*d, dtype=tf.float32), [b, d])
a = tf.reshape(tf.range(d*d, dtype=tf.float32), [d, d])

alpha = tf.matmul(tf.reshape(x, [-1, d]), a)# b*n, d
alpha = tf.matmul(tf.reshape(alpha, [b, n, d]), tf.reshape(e, [b, d, 1]))


with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  e, x , y, ans= session.run([x, e, a, alpha])
  print('*' * 10)
  print(e)
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(y)
  print('*' * 10)
  print(ans)