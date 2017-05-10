import tensorflow as tf
n = 2
r = 3

ini = []
for i in range(n):
  ini.append([j for j in range(r)])

x = tf.constant(ini,dtype=tf.float32)

sx = tf.nn.softmax(x)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())

  x, sx = session.run([x, sx])
  print(x)
  print('*' * 10)
  print(sx)
  print('*' * 10)
