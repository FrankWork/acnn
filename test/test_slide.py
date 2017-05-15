import tensorflow as tf

b = 2
n = 4
d = 5
k = 3
hk = k // 2

x = tf.reshape(tf.range(b*n*d, dtype=tf.float32), [b, n, d])
# x = tf.truncated_normal([b, n, d])
x_pad = tf.pad(x, [[0,0], [hk, hk], [0, 0]], "CONSTANT")
x_k = tf.map_fn(lambda i: x_pad[:, i:i+k, :], tf.range(n), dtype=tf.float32)# (n, b, k, d)
# list of n tensors, each shape: (b, k, d)
x_k = tf.unstack(x_k)
# axis=0: (n,b,k,d)
# axis=1: (b,n,k,d)
x_k = tf.stack(x_k, axis=1)
# x_k = x_k[0]
# print(x_k.get_shape())
# exit()
# x_k = tf.reshape(x_k, [b,n, k*d])



with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  x, xk = session.run([x, x_k])
  print('*' * 10)
  print(x.shape)
  print(x)
  print('*' * 10)
  # print(len(xk))
  # print(xk[0].shape)
  print(xk.shape)
  print(xk)