import tensorflow as tf

# 05-10 12:23 Epoch: 1 Train: 10.80% Test: 24.56%
# 05-10 12:24 Epoch: 2 Train: 16.48% Test: 32.78%
# 05-10 12:24 Epoch: 3 Train: 21.68% Test: 31.11%
# 05-10 12:24 Epoch: 4 Train: 24.59% Test: 35.07%
# 05-10 12:25 Epoch: 5 Train: 23.64% Test: 36.67%
# 05-10 12:25 Epoch: 6 Train: 24.12% Test: 32.07%
# 05-10 12:25 Epoch: 7 Train: 23.80% Test: 32.85%
# 05-10 12:25 Epoch: 8 Train: 25.87% Test: 31.26%
# 05-10 12:26 Epoch: 9 Train: 25.35% Test: 30.85%
# 05-10 12:26 Epoch: 10 Train: 25.27% Test: 32.00%
# 05-10 12:26 Epoch: 11 Train: 26.89% Test: 36.56%
# 05-10 12:27 Epoch: 12 Train: 26.91% Test: 34.11%
# 05-10 12:27 Epoch: 13 Train: 27.68% Test: 35.07%
# 05-10 12:27 Epoch: 14 Train: 27.07% Test: 36.04%
# 05-10 12:27 Epoch: 15 Train: 27.79% Test: 34.93%
# 05-10 12:28 Epoch: 16 Train: 30.09% Test: 36.56%
# 05-10 12:28 Epoch: 17 Train: 30.16% Test: 37.15%
# 05-10 12:28 Epoch: 18 Train: 30.85% Test: 40.04%
# 05-10 12:29 Epoch: 19 Train: 31.96% Test: 38.15%
# 05-10 12:29 Epoch: 20 Train: 33.71% Test: 42.89%
# 05-10 12:29 Epoch: 21 Train: 35.09% Test: 39.56%
# 05-10 12:29 Epoch: 22 Train: 36.33% Test: 37.59%
# 05-10 12:30 Epoch: 23 Train: 36.50% Test: 44.48%
# 05-10 12:30 Epoch: 24 Train: 36.90% Test: 41.63%
# 05-10 12:30 Epoch: 25 Train: 37.65% Test: 43.44%
# 05-10 12:30 Epoch: 26 Train: 38.52% Test: 49.81%
# 05-10 12:31 Epoch: 27 Train: 39.04% Test: 49.41%
# 05-10 12:31 Epoch: 28 Train: 40.67% Test: 47.30%
# 05-10 12:31 Epoch: 29 Train: 42.31% Test: 50.63%
# 05-10 12:32 Epoch: 30 Train: 42.76% Test: 50.63%
# 05-10 12:32 Epoch: 31 Train: 44.11% Test: 49.81%
# 05-10 12:32 Epoch: 32 Train: 45.09% Test: 53.63%
# 05-10 12:32 Epoch: 33 Train: 45.60% Test: 52.19%
# 05-10 12:33 Epoch: 34 Train: 46.21% Test: 53.78%
# 05-10 12:33 Epoch: 35 Train: 47.86% Test: 51.48%
# 05-10 12:33 Epoch: 36 Train: 47.95% Test: 56.89%
# 05-10 12:34 Epoch: 37 Train: 48.99% Test: 56.67%
# 05-10 12:34 Epoch: 38 Train: 49.66% Test: 56.07%
# 05-10 12:34 Epoch: 39 Train: 49.26% Test: 55.85%
# 05-10 12:34 Epoch: 40 Train: 51.26% Test: 57.19%
# 05-10 12:35 Epoch: 41 Train: 51.01% Test: 55.63%
# 05-10 12:35 Epoch: 42 Train: 51.60% Test: 58.00%
# 05-10 12:35 Epoch: 43 Train: 52.50% Test: 58.04%
# 05-10 12:36 Epoch: 44 Train: 52.51% Test: 57.85%
# 05-10 12:36 Epoch: 45 Train: 53.41% Test: 58.41%
# 05-10 12:36 Epoch: 46 Train: 53.79% Test: 57.89%
# 05-10 12:36 Epoch: 47 Train: 54.26% Test: 61.74%
# 05-10 12:37 Epoch: 48 Train: 54.19% Test: 58.56%
# 05-10 12:37 Epoch: 49 Train: 55.64% Test: 59.89%
# 05-10 12:37 Epoch: 50 Train: 55.49% Test: 60.78%
# 05-10 12:37 Epoch: 51 Train: 56.21% Test: 60.48%
# 05-10 12:38 Epoch: 52 Train: 56.66% Test: 61.04%
# 05-10 12:38 Epoch: 53 Train: 56.50% Test: 60.85%
# 05-10 12:38 Epoch: 54 Train: 57.50% Test: 61.78%
# 05-10 12:39 Epoch: 55 Train: 58.20% Test: 60.59%
# 05-10 12:39 Epoch: 56 Train: 58.05% Test: 62.11%
# 05-10 12:39 Epoch: 57 Train: 58.40% Test: 62.04%
# 05-10 12:39 Epoch: 58 Train: 57.99% Test: 62.26%
# 05-10 12:40 Epoch: 59 Train: 59.01% Test: 63.07%
# 05-10 12:40 Epoch: 60 Train: 58.90% Test: 64.33%
# 05-10 12:40 Epoch: 61 Train: 58.94% Test: 62.67%
# 05-10 12:41 Epoch: 62 Train: 60.91% Test: 64.26%
# 05-10 12:41 Epoch: 63 Train: 61.14% Test: 64.22%
# 05-10 12:41 Epoch: 64 Train: 60.72% Test: 63.93%
# 05-10 12:41 Epoch: 65 Train: 60.81% Test: 63.96%
# 05-10 12:42 Epoch: 66 Train: 62.06% Test: 63.78%
# 05-10 12:42 Epoch: 67 Train: 61.21% Test: 64.48%
# 05-10 12:42 Epoch: 68 Train: 62.69% Test: 64.30%
# 05-10 12:43 Epoch: 69 Train: 62.69% Test: 62.93%
# 05-10 12:43 Epoch: 70 Train: 62.24% Test: 63.89%
# 05-10 12:43 Epoch: 71 Train: 63.26% Test: 64.33%
# 05-10 12:43 Epoch: 72 Train: 62.70% Test: 64.93%
# 05-10 12:44 Epoch: 73 Train: 63.58% Test: 64.41%
# 05-10 12:44 Epoch: 74 Train: 64.49% Test: 65.15%
# 05-10 12:44 Epoch: 75 Train: 64.56% Test: 64.74%
# 05-10 12:45 Epoch: 76 Train: 65.34% Test: 64.33%
# 05-10 12:45 Epoch: 77 Train: 65.35% Test: 64.89%
# 05-10 12:45 Epoch: 78 Train: 65.50% Test: 64.67%
# 05-10 12:45 Epoch: 79 Train: 65.26% Test: 64.30%
# 05-10 12:46 Epoch: 80 Train: 65.56% Test: 65.59%
# 05-10 12:46 Epoch: 81 Train: 66.27% Test: 66.33%
# 05-10 12:46 Epoch: 82 Train: 66.54% Test: 64.48%
# 05-10 12:46 Epoch: 83 Train: 67.20% Test: 66.37%
# 05-10 12:47 Epoch: 84 Train: 66.83% Test: 64.78%
# 05-10 12:47 Epoch: 85 Train: 67.85% Test: 65.33%
# 05-10 12:47 Epoch: 86 Train: 67.92% Test: 64.33%
# 05-10 12:48 Epoch: 87 Train: 68.20% Test: 65.96%
# 05-10 12:48 Epoch: 88 Train: 68.45% Test: 65.52%
# 05-10 12:48 Epoch: 89 Train: 68.34% Test: 63.63%
# 05-10 12:48 Epoch: 90 Train: 68.90% Test: 65.48%
# 05-10 12:49 Epoch: 91 Train: 69.75% Test: 63.63%
# 05-10 12:49 Epoch: 92 Train: 68.42% Test: 65.93%
# 05-10 12:49 Epoch: 93 Train: 69.60% Test: 65.04%
# 05-10 12:50 Epoch: 94 Train: 69.94% Test: 65.81%
# 05-10 12:50 Epoch: 95 Train: 69.86% Test: 63.85%
# 05-10 12:50 Epoch: 96 Train: 69.86% Test: 65.30%
# 05-10 12:50 Epoch: 97 Train: 70.50% Test: 66.48%
# 05-10 12:51 Epoch: 98 Train: 71.16% Test: 65.89%
# 05-10 12:51 Epoch: 99 Train: 71.31% Test: 65.81%
# 05-10 12:51 Epoch: 100 Train: 71.06% Test: 64.00%



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
      all_distance = tf.norm(wo_norm_tile - tf.nn.l2_normalize(rel_embed, dim=1), axis=-1)

      predict = tf.argmin(all_distance, axis=1)
      predict = tf.cast(predict, dtype=tf.int32)
      acc = tf.reduce_sum(tf.cast(tf.equal(predict, in_y), dtype=tf.int32))
      self.predict = predict
      self.acc = acc

    if not is_training:
      return
      
    with tf.name_scope('loss'):
      mask = tf.one_hot(in_y, nr, on_value=1000., off_value=0.)# bz, nr
      neg_distance = tf.reduce_min(tf.add(all_distance, mask),1)
      pos_distance = tf.norm(wo_norm - tf.nn.l2_normalize(y, dim=1), axis=1)

      loss = tf.reduce_mean(pos_distance + (config.margin - neg_distance))

      l2_loss = tf.nn.l2_loss(rel_embed)
      l2_loss += tf.nn.l2_loss(U)
      l2_loss += tf.nn.l2_loss(w)
      l2_loss += tf.nn.l2_loss(b)
      l2_loss = 0.003 * config.l2_reg_lambda * l2_loss

      self.loss = loss + l2_loss

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