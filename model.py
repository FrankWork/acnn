import tensorflow as tf

# 05-10 10:22 Epoch: 1 Train: 11.03% Test: 24.07%
# 05-10 10:22 Epoch: 2 Train: 17.60% Test: 29.96%
# 05-10 10:23 Epoch: 3 Train: 23.00% Test: 33.70%
# 05-10 10:23 Epoch: 4 Train: 24.31% Test: 26.74%
# 05-10 10:23 Epoch: 5 Train: 23.30% Test: 34.33%
# 05-10 10:24 Epoch: 6 Train: 23.81% Test: 33.63%
# 05-10 10:24 Epoch: 7 Train: 24.98% Test: 32.41%
# 05-10 10:24 Epoch: 8 Train: 24.94% Test: 35.15%
# 05-10 10:24 Epoch: 9 Train: 25.56% Test: 29.78%
# 05-10 10:25 Epoch: 10 Train: 26.99% Test: 35.04%
# 05-10 10:25 Epoch: 11 Train: 26.36% Test: 35.89%
# 05-10 10:25 Epoch: 12 Train: 27.12% Test: 33.15%
# 05-10 10:26 Epoch: 13 Train: 27.02% Test: 36.67%
# 05-10 10:26 Epoch: 14 Train: 27.81% Test: 30.52%
# 05-10 10:26 Epoch: 15 Train: 28.11% Test: 31.37%
# 05-10 10:26 Epoch: 16 Train: 28.98% Test: 37.74%
# 05-10 10:27 Epoch: 17 Train: 28.64% Test: 37.04%
# 05-10 10:27 Epoch: 18 Train: 30.74% Test: 39.48%
# 05-10 10:27 Epoch: 19 Train: 30.10% Test: 34.52%
# 05-10 10:28 Epoch: 20 Train: 30.21% Test: 33.74%
# 05-10 10:28 Epoch: 21 Train: 30.68% Test: 40.00%
# 05-10 10:28 Epoch: 22 Train: 30.63% Test: 34.52%
# 05-10 10:28 Epoch: 23 Train: 31.65% Test: 43.70%
# 05-10 10:29 Epoch: 24 Train: 32.57% Test: 40.89%
# 05-10 10:29 Epoch: 25 Train: 34.30% Test: 42.30%
# 05-10 10:29 Epoch: 26 Train: 35.14% Test: 44.26%
# 05-10 10:29 Epoch: 27 Train: 36.90% Test: 45.81%
# 05-10 10:30 Epoch: 28 Train: 36.79% Test: 49.04%
# 05-10 10:30 Epoch: 29 Train: 38.66% Test: 45.56%
# 05-10 10:30 Epoch: 30 Train: 39.64% Test: 50.30%
# 05-10 10:31 Epoch: 31 Train: 41.31% Test: 52.30%
# 05-10 10:31 Epoch: 32 Train: 42.99% Test: 51.00%
# 05-10 10:31 Epoch: 33 Train: 44.36% Test: 51.00%
# 05-10 10:31 Epoch: 34 Train: 45.15% Test: 53.85%
# 05-10 10:32 Epoch: 35 Train: 45.94% Test: 54.33%
# 05-10 10:32 Epoch: 36 Train: 46.12% Test: 53.22%
# 05-10 10:32 Epoch: 37 Train: 47.21% Test: 55.07%
# 05-10 10:33 Epoch: 38 Train: 48.27% Test: 54.48%
# 05-10 10:33 Epoch: 39 Train: 49.66% Test: 57.30%
# 05-10 10:33 Epoch: 40 Train: 49.84% Test: 57.15%
# 05-10 10:33 Epoch: 41 Train: 50.22% Test: 57.04%
# 05-10 10:34 Epoch: 42 Train: 50.04% Test: 56.41%
# 05-10 10:34 Epoch: 43 Train: 51.11% Test: 58.44%
# 05-10 10:34 Epoch: 44 Train: 51.74% Test: 57.37%
# 05-10 10:35 Epoch: 45 Train: 52.84% Test: 58.59%
# 05-10 10:35 Epoch: 46 Train: 52.78% Test: 57.44%
# 05-10 10:35 Epoch: 47 Train: 52.85% Test: 56.04%
# 05-10 10:35 Epoch: 48 Train: 54.60% Test: 61.04%
# 05-10 10:36 Epoch: 49 Train: 54.95% Test: 60.07%
# 05-10 10:36 Epoch: 50 Train: 54.95% Test: 59.81%
# 05-10 10:36 Epoch: 51 Train: 55.01% Test: 59.04%
# 05-10 10:37 Epoch: 52 Train: 56.17% Test: 60.26%
# 05-10 10:37 Epoch: 53 Train: 56.26% Test: 62.30%
# 05-10 10:37 Epoch: 54 Train: 56.79% Test: 61.85%
# 05-10 10:37 Epoch: 55 Train: 57.67% Test: 62.52%
# 05-10 10:38 Epoch: 56 Train: 57.94% Test: 61.70%
# 05-10 10:38 Epoch: 57 Train: 58.07% Test: 62.52%
# 05-10 10:38 Epoch: 58 Train: 58.34% Test: 61.89%
# 05-10 10:39 Epoch: 59 Train: 59.41% Test: 62.48%
# 05-10 10:39 Epoch: 60 Train: 60.12% Test: 62.70%

class Model(object):
  def __init__(self, config, embeddings, is_training=True):
    bz = config.batch_size
    dw = config.embedding_size
    dp = config.pos_embed_size
    d = dw+2*dp
    np = config.pos_embed_num
    n = config.max_len
    k = config.slide_window
    dc = config.num_filters
    nr = config.classnum # number of relations
    keep_prob = config.keep_prob

    with tf.name_scope('input'):
      in_x = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_x') # sentences
      in_e1 = tf.placeholder(dtype=tf.int32, shape=[bz], name='in_e1')
      in_e2 = tf.placeholder(dtype=tf.int32, shape=[bz], name='in_e2')
      in_dist1 = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_dist1')
      in_dist2 = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_dist2')
      in_y = tf.placeholder(dtype=tf.int32, shape=[bz], name='in_y') # relations
      
      self.inputs = (in_x, in_e1, in_e2, in_dist1, in_dist2, in_y)
    
    with tf.name_scope('embeddings'):
      initializer = tf.truncated_normal_initializer(stddev=0.1)
      embed = tf.get_variable(initializer=embeddings, dtype=tf.float32, name='word_embed')
      pos1_embed = tf.get_variable(shape=[np, dp],name='position1_embed')
      pos2_embed = tf.get_variable(shape=[np, dp],name='position2_embed')
      # pos1_embed = tf.get_variable(initializer=initializer,shape=[np, dp],name='position1_embed')
      # pos2_embed = tf.get_variable(initializer=initializer,shape=[np, dp],name='position2_embed')
      rel_embed = tf.get_variable(initializer=initializer,shape=[nr, dc],name='relation_embed')

      # embdding lookup
      e1 = tf.nn.embedding_lookup(embed, in_e1, name='e1')# bz,dw
      e2 = tf.nn.embedding_lookup(embed, in_e2, name='e2')# bz,dw
      x = tf.nn.embedding_lookup(embed, in_x, name='x')   # bz,n,dw
      dist1 = tf.nn.embedding_lookup(pos1_embed, in_dist1, name='dist1')#bz, n, k,dp
      dist2 = tf.nn.embedding_lookup(pos2_embed, in_dist2, name='dist2')# bz, n, k,dp
      y = tf.nn.embedding_lookup(rel_embed, in_y, name='y')# bz, dc

      x_conv = tf.reshape(tf.concat([x, dist1, dist2], -1), # bz, n, d
                        [bz,n,d,1])
      if is_training and keep_prob < 1:
        x_conv = tf.nn.dropout(x_conv, keep_prob)


    with tf.name_scope('input_attention'):
      A1 = tf.matmul(x, tf.expand_dims(e1, -1))
      A2 = tf.matmul(x, tf.expand_dims(e2, -1))
      A1 = tf.reshape(A1, [bz, n])
      A2 = tf.reshape(A2, [bz, n])
      alpha1 = tf.nn.softmax(A1)# bz, n
      alpha2 = tf.nn.softmax(A2)# bz, n
      alpha = (alpha1 + alpha2)/2
    
    
    

    with tf.name_scope('convolution'):
      # x: (batch_size, max_len, embdding_size, 1)
      # w: (filter_size, embdding_size, 1, num_filters)
      w = tf.get_variable(initializer=initializer,shape=[k, d, 1, dc],name='weight')
      b = tf.get_variable(initializer=initializer,shape=[dc],name='bias')
      conv = tf.nn.conv2d(x_conv, w, strides=[1,1,d,1],padding="SAME")
      r = conv
      # r = tf.multiply(tf.reshape(conv, [bz, n, dc]), tf.reshape(alpha, [bz, n, 1])) # bz, n, 1, dc
      
      
      R = tf.nn.tanh(tf.nn.bias_add(r,b),name="R") # bz, n, 1, dc
      R = tf.reshape(R, [bz, n, dc])







    with tf.name_scope('attention_pooling'):
      U = tf.get_variable(initializer=initializer,shape=[dc,nr],name='U')
      G = tf.matmul(# (bz*n,dc), (dc, nr) => (bz*n, nr)
        tf.reshape(R, [bz*n, dc]), U
      )
      G = tf.matmul(# (bz*n, nr), (nr, dc) => (bz*n, dc)
        G, rel_embed
      ) 
      G = tf.reshape(G, [bz, n, dc])
      AP = tf.nn.softmax(G, dim=1)# attention pooling tensor
      # predict
      wo = tf.matmul(
        tf.transpose(R, perm=[0, 2, 1]), # batch transpose: (bz, n, dc) => (bz,dc,n)
        AP
      )# (bz, dc, dc)
      # wo = tf.reduce_max(wo, axis=-1) # (bz, dc)
      wo = tf.nn.max_pool(tf.expand_dims(wo,-1),
                          ksize=[1,1,dc,1],
                          strides=[1,1,dc,1],
                          padding="SAME"
            )# (bz, dc, 1, 1)
      wo=tf.reshape(wo,[bz, dc])






      # # attention pooling
      # U = tf.get_variable(initializer=initializer,shape=[dc,dc],name='U')
      
      # # batch matmul
      # # G = R * U * WL
      # # R: (bz, n, dc)
      # # U: (dc, dc)
      # # WL:(dc, nr)
      # G = tf.matmul(# (bz*n,dc), (dc, dc) => (bz*n, dc)
      #   tf.reshape(R, [bz*n, dc]), U
      # )
      # G = tf.matmul(# (bz*n, dc), (dc, nr) => (bz*n, nr)
      #   G, tf.transpose(rel_embed)
      # ) 
      # G = tf.reshape(G, [bz, n, nr])
      # AP = tf.nn.softmax(G, dim=1)# attention pooling tensor

      # # predict
      # wo = tf.matmul(
      #   tf.transpose(R, perm=[0, 2, 1]), # batch transpose: (bz, n, dc) => (bz,dc,n)
      #   AP
      # )# (bz, dc, nr)
      # wo = tf.reduce_max(wo, axis=-1) # (bz, dc)
      # # wo = tf.nn.max_pool(tf.expand_dims(wo,-1),
      # #                     ksize=[1,1,nr,1],
      # #                     strides=[1,1,nr,1],
      # #                     padding="SAME"
      # #       )# (bz, dc, 1, 1)
      # # wo=tf.reshape(wo,[bz, dc])

      if is_training and keep_prob < 1:
        wo = tf.nn.dropout(wo, keep_prob)

    with tf.name_scope('predict'):
      wo_norm = tf.nn.l2_normalize(wo, 1)
      wo_norm_tile = tf.tile(tf.expand_dims(wo_norm, axis=1), [1, nr, 1])
      all_distance = wo_norm_tile - tf.nn.l2_normalize(rel_embed, dim=1)
      all_distance = tf.sqrt(tf.reduce_sum(tf.square(all_distance), 2))

      predict = tf.argmin(all_distance, axis=1)
      predict = tf.cast(predict, dtype=tf.int32)
      acc = tf.reduce_sum(tf.cast(tf.equal(predict, in_y), dtype=tf.int32))
      self.predict = predict
      self.acc = acc




      # # accuracy
      # def distance(wo, y, axis=None):
      #   return tf.norm(
      #             tf.nn.l2_normalize(wo, dim=-1) - tf.nn.l2_normalize(y,dim=-1),
      #             axis = axis
      #         )# a scalar value


      # all_dist = distance(
      #           tf.tile(tf.expand_dims(wo, axis=1), [1, nr, 1]), # bz, nr, dc
      #           rel_embed,# nr, dc
      #           axis=-1
      # )# bz, nr
      # predict = tf.argmin(all_dist, axis=-1)
      # predict = tf.cast(predict, dtype=tf.int32)
      # acc = tf.reduce_sum(tf.cast(tf.equal(predict, in_y), dtype=tf.int32))
      # self.predict = predict
      # self.acc = acc


    if not is_training:
      return
      
    with tf.name_scope('loss'):
      mask = tf.one_hot(in_y, nr, on_value=1000., off_value=0.)# bz, nr
      neg_distance = tf.reduce_min(tf.add(all_distance, mask),1)

      pos_distance = wo_norm - tf.nn.l2_normalize(y, dim=1)
      pos_distance = tf.sqrt(tf.reduce_sum(tf.square(pos_distance),1))

      loss = tf.reduce_mean(pos_distance + (config.margin - neg_distance))

      l2_loss = tf.nn.l2_loss(rel_embed)
      l2_loss += tf.nn.l2_loss(U)
      l2_loss += tf.nn.l2_loss(w)
      l2_loss += tf.nn.l2_loss(b)
      l2_loss = 0.003 * config.l2_reg_lambda * l2_loss

      self.loss = loss + l2_loss




      # mask = tf.one_hot(in_y, nr, on_value=1000., off_value=0.)# bz, nr
      # neg_dist = tf.reduce_min(tf.add(all_dist, mask), 1)
      # neg_dist = tf.reduce_mean(neg_dist)
      # loss = distance(wo, y) + (config.margin - neg_dist)

      # l2_loss = tf.nn.l2_loss(rel_embed)
      # l2_loss += tf.nn.l2_loss(U)
      # l2_loss += tf.nn.l2_loss(w)
      # l2_loss += tf.nn.l2_loss(b)
      # l2_loss = 0.003 * config.l2_reg_lambda * l2_loss

      # self.loss = loss + l2_loss






    with tf.name_scope('optimizer'):
      # optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
      optimizer = tf.train.AdamOptimizer(config.learning_rate)
      # optimizer2 = tf.train.AdamOptimizer(config.learning_rate2)

      # tvars = tf.trainable_variables()
      # grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
      #                                   config.grad_clipping)
      # capped_gvs = zip(grads, tvars)

      # tf.logging.set_verbosity(tf.logging.ERROR)
      global_step = tf.Variable(0, trainable=False, name='global_step')
      # train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
      # reg_op = optimizer2.minimize(l2_loss)



      self.train_op = optimizer.minimize(self.loss)
      # self.reg_op = reg_op
      self.reg_op = tf.no_op()
      self.global_step = global_step