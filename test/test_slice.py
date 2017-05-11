import tensorflow as tf

b = 2
n = 10




x = tf.reshape(tf.range(1, b*n+1), [b, n])
e = tf.range(b)

# y = tf.slice()

with tf.Session() as session:
  session.run(tf.global_variables_initializer())

  x, e = session.run([x, e])
  print(x)
  print(e)



