import tensorflow as tf

b = 1
n = 4
d = 3
k = 3

dc = 3

x = tf.reshape(tf.range(b*n*dc, dtype=tf.float32), [b, n, dc])
a = tf.reshape(tf.range(b*n, dtype=tf.float32), [b, n])
y = tf.multiply(x, tf.reshape(a, [b, n, 1]))

y2 =tf.matmul(
        tf.tile(tf.reshape(a,[-1,1, n]),[1,n,1]), # bz, n, n
        x # bz, n, dc
      )


with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  x, a, y, y2 = session.run([x, a, y, y2])
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(a)
  print('*' * 10)
  print(y)
  print('*' * 10)
  print(y2)
