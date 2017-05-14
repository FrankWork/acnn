import tensorflow as tf

b = 2
n = 3
d = 3
k = 3
hk = k // 2
dc = 3

x = tf.reshape(tf.range(b*n*d, dtype=tf.float32), [b, n, d])
x_pad = tf.pad(x, [[0,0], [hk, hk], [0, 0]], "CONSTANT")
x_k = tf.map_fn(lambda i: x_pad[:, i:i+k, :], tf.range(n), dtype=tf.float32)
x_k = tf.stack(tf.unstack(x_k), axis=2)
# x_k = tf.reshape(x_k, [b,n, k*d])

w = tf.get_variable(initializer=tf.reshape(tf.range(k*d*dc, dtype=tf.float32),shape=[k*d, dc]),name='weight')

y = tf.matmul(tf.reshape(x_k, [b*n, k*d]),w)
y = tf.reshape(y,[b, n, dc])

g = tf.gradients(y, tf.trainable_variables())

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  x, w, y, g = session.run([x, w, y, g])
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(w)
  print('*' * 10)
  print(y)
  print('*' * 10)
  print(g)

# [[[  300.   315.   330.]
#   [  612.   648.   684.]
#   [  300.   333.   366.]]

#  [[ 1191.  1260.  1329.]
#   [ 1584.  1701.  1818.]
#   [  705.   792.   879.]]]

# **********
# [array([[ 24.,  24.,  24.],
#        [ 28.,  28.,  28.],
#        [ 32.,  32.,  32.],
#        [ 45.,  45.,  45.],
#        [ 51.,  51.,  51.],
#        [ 57.,  57.,  57.],
#        [ 36.,  36.,  36.],
#        [ 40.,  40.,  40.],
#        [ 44.,  44.,  44.]], dtype=float32)]

