import tensorflow as tf

b = 2
n = 3
d = 4
k = 3

dc = 5

x = tf.reshape(tf.range(b*n*d, dtype=tf.float32), [b, n, d])

w = tf.get_variable(initializer=tf.ones_initializer(),shape=[k, d, 1, dc],name='weight')
# b = tf.get_variable(initializer=initializer,shape=[dc],name='bias')
conv = tf.nn.conv2d(tf.reshape(x, # bz, n, d
                          [b,n,d,1]), w, strides=[1,1,d,1], padding="SAME")
# b, n, 1, dc
# r = tf.multiply(tf.reshape(conv, [b, n, dc]), tf.reshape(alpha, [b, n, 1])) # b, n, 1, dc

# R = tf.nn.tanh(tf.nn.bias_add(r,b),name="R") # b, n, 1, dc
# R = tf.reshape(R, [b, n, dc])

y = conv

ans = tf.reduce_sum(tf.multiply(
  x[0],
  tf.ones([k,d])
))
# ans = tf.constant(0)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  e, x, y, ans = session.run([w, x, y, ans])
  print('*' * 10)
  print(e)
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(y)
  print('*' * 10)
  print(ans)