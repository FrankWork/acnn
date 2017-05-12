import tensorflow as tf

b = 2
n = 10

def slide_window(x, k):
  hk = k // 2 # half k
  x_pad = tf.pad(x, [[0,0], [hk,hk]], "CONSTANT")# bz, n+2*(k-1)
  x_k = tf.map_fn(lambda i: x_pad[:, i:i+k], tf.range(n), dtype=tf.int32)
  return tf.stack(tf.unstack(x_k), axis=1)# bz, n, k



x = tf.reshape(tf.range(1, b*n+1), [b, n])
# xm = tf.map_fn(lambda i: x[i:i+4], tf.range(20-4+1), dtype=tf.int32)
xm = slide_window(x, 4)


with tf.Session() as session:
  session.run(tf.global_variables_initializer())

  x, xm = session.run([x, xm])
  print(x)
  print(xm)
