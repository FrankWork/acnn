import tensorflow as tf

b = 2
n = 3
r = 4


x = tf.reshape(tf.range(b*n*r, dtype=tf.float32), [b, n, r])*10
y = tf.nn.softmax(x, dim=1)

ans = tf.nn.softmax(
  tf.convert_to_tensor([0, 40, 80],dtype=tf.float32)
)

with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  e, x , ans= session.run([x, y, ans])
  print('*' * 10)
  print(e)
  print('*' * 10)
  print(x)
  print('*' * 10)
  print(ans)
  