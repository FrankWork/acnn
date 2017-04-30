import tensorflow as tf

b = 2
n = 3
d = 4

def distance(wo, y, axis=None):
  return tf.norm(
    tf.nn.l2_normalize(wo, dim=-1) - y,
    axis = axis
  )# a scalar value

x = tf.reshape(tf.range(d, dtype=tf.float32), [d])
x_norm = tf.nn.l2_normalize(x, dim=-1)
label = tf.ones([d])

y = x_norm - label

ans = distance(x, label)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  e, x , y, ans= session.run([x, x_norm, y, ans])
  print('*' * 10)
  print(e)
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(y)
  print('*' * 10)
  print(ans)