import tensorflow as tf

b = 2
d = 4
n = 3

 # input attention
def inner_product(e, x):
  '''
  <x, y> = x1y1 + x2y2 + ... + xnyn
  e:        (bz, dw) => (bz, 1, dw)
  x:        (bz, n, dw)
  return :  (bz, n)
  '''
  return tf.reduce_sum(
            tf.multiply(tf.reshape(e, [b, 1, d]), x), 
            -1
          )

e = tf.reshape(tf.range(b*d, dtype=tf.float32), [b, d])
x = tf.reshape(tf.range(b*n*d, dtype=tf.float32), [b, n, d])
y = inner_product(e, x)

ans = tf.multiply(
  tf.reshape(tf.range(d, dtype=tf.float32), [d]),
  tf.reshape(tf.range(n*d, dtype=tf.float32), [n, d])
)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  e, x, y, ans = session.run([e, x, y, ans])
  print('*' * 10)
  print(e)
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(y)
  print('*' * 10)
  print(ans)