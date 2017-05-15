import tensorflow as tf

b = 2
n = 4
d = 5
k = 3
hk = k // 2
dc = 3

# x = tf.reshape(tf.range(b*n*d, dtype=tf.float32), [b, n, d])
x = tf.truncated_normal([b, n, d])
initializer = tf.truncated_normal_initializer(stddev=0.1)
w = tf.get_variable(initializer=initializer,shape=[k*d, dc],name='weight')

x_pad = tf.pad(x, [[0,0], [hk, hk], [0, 0]], "CONSTANT")
x_map = tf.map_fn(lambda i: x_pad[:, i:i+k, :], tf.range(n), dtype=tf.float32)
x_stack = tf.stack(tf.unstack(x_map), axis=1) # (b,n,k,d)


y = tf.matmul(tf.reshape(x_stack, [b*n, k*d]),w)
y = tf.reshape(y,[b, n, dc])


conv = tf.nn.conv2d(tf.reshape(x, [b,n,d,1]), 
                    tf.reshape(w, [k,d, 1,dc]),
                    strides=[1,1,d,1], 
                    padding="SAME")

y2 = tf.reshape(conv, [b, n, dc])

loss = tf.reduce_sum(tf.abs(y - y2))



with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  xk, y, y2, loss = session.run([x_stack, y, y2, loss])
  print('*' * 10)
  print(xk.shape)
  print(xk)
  print('*' * 10)
  print(loss)
  # print('*' * 10)
  # print(y)
  # print('*' * 10)
  # print(y2)

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

