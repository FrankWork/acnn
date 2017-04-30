import tensorflow as tf

b = 2
d = 4
n = 3

alpha = tf.reshape(tf.range(b*n, dtype=tf.float32), [b, n, 1])
x = tf.reshape(tf.range(b*n*d, dtype=tf.float32), [b, n, d])
y = tf.multiply(x, alpha)

# ans = tf.multiply(
#   tf.reshape(tf.range(d, dtype=tf.float32), [d]),
#   tf.reshape(tf.range(n*d, dtype=tf.float32), [n, d])
# )

ans = tf.constant(0)


with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  e, x, y, ans = session.run([alpha, x, y, ans])
  print('*' * 10)
  print(e)
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(y)
  print('*' * 10)
  print(ans)