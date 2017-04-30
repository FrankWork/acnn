import tensorflow as tf

b = 4
r = 3
d = 4

def distance(wo, y, axis=None):
  return tf.norm(
    tf.nn.l2_normalize(wo, dim=-1) - y,
    axis = axis
  )# a scalar value

x = tf.reshape(tf.range(b*d, dtype=tf.float32), [b, d])
x_tile = tf.tile(tf.expand_dims(x, axis=1), [1,r,1]) # b,r,d
# x_norm = tf.nn.l2_normalize(x_tile, dim=-1)
x_norm = x_tile
label = tf.reshape(tf.range(r*d, dtype=tf.float32), [r, d])

y = x_norm - label

# ans = distance(x, label)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  all = session.run([x, x_tile, x_norm, label, y])
  for v in all:
    print('*' * 10)
    print(v)