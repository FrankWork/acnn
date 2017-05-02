import tensorflow as tf

b = 2
n = 3
d = 4
k = 3

dc = 5

x = tf.reshape(tf.range(b*n*d, dtype=tf.float32), [b, n, d])

pool = tf.nn.max_pool(tf.expand_dims(x, axis=-1), 
                      ksize=[1, 1, d, 1], 
                      strides=[1, 1, d, 1],
                      padding="SAME")

pool = tf.reshape(pool, [b, n])

max = tf.reduce_max(x, axis=-1) # (bz, dc)

tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())

  x, pool, max = session.run([x, pool, max])
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(pool)
  print('*' * 10)
  print(max)