import tensorflow as tf

class Model(object):
  def __init__(self, config, embeddings, is_training=True):
    bz = config.batch_size
    n = config.max_len
    k = config.slide_window
    dw = config.embedding_size
    dp = config.pos_embed_size
    d = dw+2*dp
    np = config.pos_embed_num
    nr = config.classnum # number of relations
    dc = config.num_filters
    keep_prob = config.keep_prob
    self.config = config

    with tf.name_scope('input'):
      in_x = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_x') # sentences
      in_e1 = tf.placeholder(dtype=tf.int32, shape=[bz], name='in_e1')
      in_e2 = tf.placeholder(dtype=tf.int32, shape=[bz], name='in_e2')
      in_dist1 = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_dist1')
      in_dist2 = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_dist2')
      in_y = tf.placeholder(dtype=tf.int32, shape=[bz], name='in_y') # relations
      
      self.inputs = (in_x, in_e1, in_e2, in_dist1, in_dist2, in_y)

      initializer = tf.truncated_normal_initializer(stddev=0.1)
      # initializer = tf.ones_initializer()
    
    with tf.name_scope('embeddings'):
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

      x_concat = tf.concat([x, dist1, dist2], -1) # bz, n, d
      if is_training and keep_prob < 1:
        x_concat = tf.nn.dropout(x_concat, keep_prob)

      self.l2_loss = tf.nn.l2_loss(rel_embed)
    
    with tf.name_scope('forword'):
      alpha = self._input_attention(x, e1, e2, initializer=initializer)

      if config.standard_conv:
        R = self._standard_conv(x_concat, initializer=initializer, alpha=alpha)
      else:
        R = self._slide_conv(x_concat, initializer=initializer, alpha=alpha)

      wo = self._attentive_pooling(R, rel_embed, initializer=initializer)

      if is_training and keep_prob < 1:
        wo = tf.nn.dropout(wo, keep_prob)
      
      # self.R = R
      self._loss_and_train(wo, rel_embed, in_y, y, is_training)

    
  

  def _input_attention(self, x, e1, e2, initializer=None):
    bz = self.config.batch_size
    n = self.config.max_len

    with tf.name_scope('input_attention'):
      A1 = tf.matmul(x, tf.expand_dims(e1, -1))# bz, n, 1
      A2 = tf.matmul(x, tf.expand_dims(e2, -1))
      A1 = tf.reshape(A1, [bz, n])
      A2 = tf.reshape(A2, [bz, n])
      alpha1 = tf.nn.softmax(A1)# bz, n
      alpha2 = tf.nn.softmax(A2)# bz, n

      # bz = self.config.batch_size
      # n = self.config.max_len
      # dw = self.config.embedding_size

      # A1 = tf.get_variable(initializer=initializer,shape=[dw, dw],name='A1')
      # alpha1 = tf.matmul(tf.reshape(x, [-1, dw]), A1)# b*n, d
      # alpha1 = tf.matmul(tf.reshape(alpha1, [bz, n, dw]), tf.reshape(e1, [bz, dw, 1]))
      # alpha1 = tf.nn.softmax(tf.reshape(alpha1, [bz, n]))
      # A2 = tf.get_variable(initializer=initializer,shape=[dw, dw],name='A2')
      # alpha2 = tf.matmul(tf.reshape(x, [-1, dw]), A2)# b*n, d
      # alpha2 = tf.matmul(tf.reshape(alpha2, [bz, n, dw]), tf.reshape(e2, [bz, dw, 1]))
      # alpha2 = tf.nn.softmax(tf.reshape(alpha2, [bz, n]))

      # self.l2_loss += tf.nn.l2_loss(A1)
      # self.l2_loss += tf.nn.l2_loss(A2)

      
      alpha = (alpha1 + alpha2)/2
      
      return alpha

  def _slide_conv(self, x_concat, initializer=None, alpha=None):
    bz = self.config.batch_size
    n = self.config.max_len
    k = self.config.slide_window
    dw = self.config.embedding_size
    dp = self.config.pos_embed_size
    d = dw+2*dp
    dc = self.config.num_filters

    with tf.variable_scope('slide_conv'):
      # conv with explicit slide window

      # slide window
      hk = k // 2 # half k
      x_pad = tf.pad(x_concat, [[0,0], [hk, hk], [0, 0]], "CONSTANT")
      x_k = tf.map_fn(lambda i: x_pad[:, i:i+k, :], tf.range(n), dtype=tf.float32)
      x_k = tf.stack(tf.unstack(x_k), axis=1)
      x_concat = tf.reshape(x_k, [bz,n, k*d])

      x_concat = tf.multiply(x_concat, tf.reshape(alpha, [bz, n, 1])) # bz, n, k*d
      w = tf.get_variable(initializer=initializer,shape=[k*d, dc],name='weight')
      b = tf.get_variable(initializer=initializer,shape=[dc],name='bias')
      conv = tf.matmul(tf.reshape(x_concat, [bz*n, k*d]), w)
      R = tf.nn.tanh(tf.nn.bias_add(conv,b),name="R") # bz*n, dc
      R = tf.reshape(R, [bz, n, dc])
      
    
    self.l2_loss += tf.nn.l2_loss(w)
    self.l2_loss += tf.nn.l2_loss(b)
    return R

  def _standard_conv(self, x_concat, initializer=None, alpha=None):
    bz = self.config.batch_size
    n = self.config.max_len
    k = self.config.slide_window
    dw = self.config.embedding_size
    dp = self.config.pos_embed_size
    d = dw+2*dp
    dc = self.config.num_filters
    with tf.variable_scope('std_conv'):
      # x: (batch_size, max_len, embdding_size, 1)
      # w: (filter_size, embdding_size, 1, num_filters)
      w = tf.get_variable(initializer=initializer,shape=[k, d, 1, dc],name='weight')
      b = tf.get_variable(initializer=initializer,shape=[dc],name='bias')
      x_concat = tf.reshape( x_concat, [bz,n,d,1])
      conv = tf.nn.conv2d(x_concat, w, strides=[1,1,d,1],padding="SAME")# bz, n, 1, dc
      R = tf.nn.tanh(tf.nn.bias_add(conv,b),name="R") # bz, n, 1, dc

      R = tf.reshape(R, [bz, n, dc])
      R = tf.multiply(R, tf.reshape(alpha, [bz, n, 1])) # bz, n, dc
    self.l2_loss += tf.nn.l2_loss(w)
    self.l2_loss += tf.nn.l2_loss(b)
    return R

  
  def _attentive_pooling(self, R, rel_embed, initializer=None):
    bz = self.config.batch_size
    n = self.config.max_len
    k = self.config.slide_window
    dw = self.config.embedding_size
    dp = self.config.pos_embed_size
    d = dw+2*dp
    dc = self.config.num_filters
    nr = self.config.classnum

    with tf.name_scope('attention_pooling'):
      # # no attention_pooling
      # wo = tf.nn.max_pool(tf.expand_dims(R,-1),# bz, n, dc, 1
      #                     ksize=[1,n,1,1],
      #                     strides=[1,n,1,1],
      #                     padding="SAME"
      #       )# (bz, 1, dc, 1)
      # wo=tf.reshape(wo,[bz, dc])
      # W_o = tf.get_variable(initializer=initializer,shape=[dc, dc],name='w_o')
      # b_o = tf.get_variable(initializer=initializer,shape=[dc],name='b_o')
      # wo = tf.nn.xw_plus_b(wo,W_o,b_o,name="scores")
      

      # U: [dc, nr]
      U = tf.get_variable(initializer=initializer,shape=[dc,nr],name='U')
      G = tf.matmul(tf.reshape(R, [bz*n, dc]), U)# (bz*n,dc), (dc, nr) => (bz*n, nr)
      G = tf.matmul(G, rel_embed) # (bz*n, nr), (nr, dc) => (bz*n, dc)
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
      
      self.l2_loss += tf.nn.l2_loss(U)
      # self.l2_loss += tf.nn.l2_loss(W_o)
      # self.l2_loss += tf.nn.l2_loss(b_o)
      
    return wo

  def _loss_and_train(self, wo, rel_embed, in_y, y, is_training):
    nr = self.config.classnum

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

      loss = tf.reduce_mean(pos_distance + (self.config.margin - neg_distance))
      self.loss = loss + 0.003 * self.config.l2_reg_lambda * self.l2_loss

    with tf.name_scope('optimizer'):
      # optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
      optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
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