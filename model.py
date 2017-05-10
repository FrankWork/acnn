import tensorflow as tf

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
      # # U: [dc, nr]
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

      # # U: [dc, dc]
      # U = tf.get_variable(initializer=initializer,shape=[dc,dc],name='U')
      # G = tf.matmul(tf.reshape(R, [bz*n, dc]), U)
      # G = tf.matmul(G, tf.transpose(rel_embed)) 
      # G = tf.reshape(G, [bz, n, nr])
      # AP = tf.nn.softmax(G, dim=1)# attention pooling tensor

      # wo = tf.matmul(tf.transpose(R, perm=[0, 2, 1]),AP)# (bz, dc, nr)
      # wo = tf.reduce_max(wo, axis=-1) # (bz, dc)

      if is_training and keep_prob < 1:
        wo = tf.nn.dropout(wo, keep_prob)

    with tf.name_scope('predict'):
      wo_norm = tf.nn.l2_normalize(wo, 1)
      wo_norm_tile = tf.tile(tf.expand_dims(wo_norm, axis=1), [1, nr, 1])
      all_distance = tf.norm(wo_norm_tile - tf.nn.l2_normalize(rel_embed, dim=1), axis=2)

      predict = tf.argmin(all_distance, axis=1)
      predict = tf.cast(predict, dtype=tf.int32)
      acc = tf.reduce_sum(tf.cast(tf.equal(predict, in_y), dtype=tf.int32))
      self.predict = predict
      self.acc = acc

    if not is_training:
      return
      
    with tf.name_scope('loss'):
      mask = tf.one_hot(in_y, nr, on_value=1000., off_value=0.)# bz, nr
      # neg_distance = tf.reduce_min(tf.add(all_distance, mask),1)
      neg_y = tf.argmin(tf.add(all_distance, mask), axis=1)# bz,
      neg_y = tf.nn.embedding_lookup(rel_embed, neg_y)# bz, dc
      neg_distance = tf.norm(wo_norm - tf.nn.l2_normalize(neg_y, dim=1), axis=1)

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