import tensorflow as tf

b = 2
n = 3
d = 4

def distance(wo, y, axis=None):
  return tf.norm(
    tf.nn.l2_normalize(wo, dim=-1) - tf.nn.l2_normalize(y, dim=-1),
    axis = axis
  )# a scalar value

x = tf.reshape(tf.range(d, dtype=tf.float32), [d])
label = tf.ones([d])

ans = distance(x, label)

y = tf.nn.l2_normalize(x, -1) - tf.nn.l2_normalize(label,dim=-1)
y = tf.sqrt(tf.reduce_sum(tf.square(y),-1))

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  e, x , y, ans= session.run([x, label, y, ans])
  print('*' * 10)
  print(e)
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(y)
  print('*' * 10)
  print(ans)