import tensorflow as tf

def _bi_rnn(config, inputs, seq_len, is_training=True, scope=None):
  '''
  return value:
    output:(output_fw, output_bw) [batch_size, max_time, hidden_size]
    state: (state_fw, state_bw) ([batch_size, hidden_size], ...) len() == num_layers
  '''
  def gru_cell():
    return tf.contrib.rnn.GRUCell(config.hidden_size)
  cell = gru_cell
  if is_training and config.keep_prob < 1:
    def cell():
      return tf.contrib.rnn.DropoutWrapper(
            gru_cell(), output_keep_prob=config.keep_prob)
  cell_fw = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config.num_layers)] )
  cell_bw = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config.num_layers)] )

  return tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, 
                          sequence_length=seq_len, dtype=tf.float32, scope=scope)


class Model(object):
  def __init__(self, config, embeddings, is_training=True):
    bz = config.batch_size
    dw = config.embedding_size
    dp = config.pos_embed_size
    np = config.pos_embed_num
    n = config.max_len
    d = dw+2*dp
    hz = config.hidden_size
    # k = config.slide_window
    dc = config.num_filters
    nr = config.classnum # number of relations
    keep_prob = config.keep_prob

    # input
    in_x = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_x') # sentences
    # in_len = tf.placeholder(dtype=tf.int32, shape=[bz], name='in_len') # real length of each sentences
    in_e1 = tf.placeholder(dtype=tf.int32, shape=[bz, 3], name='in_e1')
    in_e2 = tf.placeholder(dtype=tf.int32, shape=[bz, 3], name='in_e2')
    in_dist1 = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_dist1')
    in_dist2 = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_dist2')
    in_y = tf.placeholder(dtype=tf.int32, shape=[bz], name='in_y') # relations
    
    # self.inputs = (in_x, in_len, in_e1, in_e2, in_dist1, in_dist2, in_y)
    self.inputs = (in_x, in_e1, in_e2, in_dist1, in_dist2, in_y)
    
    # embeddings
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    embed = tf.get_variable(initializer=embeddings, dtype=tf.float32, name='word_embed')
    pos1_embed = tf.get_variable(initializer=initializer,shape=[np, dp],name='position1_embed')
    pos2_embed = tf.get_variable(initializer=initializer,shape=[np, dp],name='position2_embed')
    rel_embed = tf.get_variable(initializer=initializer,shape=[nr, dc],name='relation_embed')

    # def slide_window(x, k):
    #   hk = k // 2 # half k
    #   x_pad = tf.pad(x, [[0,0], [hk,hk]], "CONSTANT")# bz, n+2*(k-1)
    #   x_k = tf.map_fn(lambda i: x_pad[:, i:i+k], tf.range(n), dtype=tf.int32)
    #   return tf.stack(tf.unstack(x_k), axis=1)# bz, n, k
    
    # x_3 = slide_window(in_x, 3)
    
    # embdding lookup
    e1 = tf.nn.embedding_lookup(embed, in_e1, name='e1')# bz,dw
    e2 = tf.nn.embedding_lookup(embed, in_e2, name='e2')# bz,dw
    x = tf.nn.embedding_lookup(embed, in_x, name='x')   # bz,n,dw
    # x_3 = tf.nn.embedding_lookup(embed, x_3, name='x_3')
    dist1 = tf.nn.embedding_lookup(pos1_embed, in_dist1, name='dist1')#bz, n, k,dp
    dist2 = tf.nn.embedding_lookup(pos2_embed, in_dist2, name='dist2')# bz, n, k,dp
    # y = tf.nn.embedding_lookup(rel_embed, in_y, name='y')# bz, dc
    x_concat = tf.concat([x, dist1, dist2], -1) # bz, n, d


    # # input attention
    # x_3 = tf.reshape(x_3, [bz, n, 3*dw])
    # A1 = tf.matmul(x_3, tf.reshape(e1, [bz, 3*dw, 1]))# bz, n, 1
    # A2 = tf.matmul(x_3, tf.reshape(e2, [bz, 3*dw, 1]))
    # A1 = tf.reshape(A1, [bz, n])
    # A2 = tf.reshape(A2, [bz, n])
    # alpha1 = tf.nn.softmax(A1)# bz, n
    # alpha2 = tf.nn.softmax(A2)# bz, n
    # alpha = (alpha1 + alpha2)/2



    # bidirectional rnn
    # output_rnn, state_rnn = _bi_rnn(config, x_concat, in_len, is_training, 'rnn')
    # x_rnn = tf.concat([output_rnn[0], output_rnn[1]], axis=2) # xi = (fw_hi, bw_hi), shape: (bz, n, hz)
    
    # d = hz # dw+2*dp => hz
    # x_concat = tf.reshape(x_rnn, [bz,n,d,1])

    x_concat = tf.reshape(x_concat, [bz,n,d,1])

    # convolution
    # x: (batch_size, max_len, embdding_size, 1)
    # w: (filter_size, embdding_size, 1, num_filters)
    filter_sizes = [3, 4, 5]
    pooled_outputs = []
    if is_training and keep_prob < 1:
      x_concat = tf.nn.dropout(x_concat, keep_prob)

    for i, k in enumerate(filter_sizes):
      with tf.variable_scope("conv-%d" % k):# , reuse=False
        w = tf.get_variable(initializer=initializer,shape=[k, d, 1, dc],name='weight')
        b = tf.get_variable(initializer=initializer,shape=[dc],name='bias')
        conv = tf.nn.conv2d(x_concat, w, strides=[1,1,d,1],padding="SAME")

        h = tf.nn.tanh(tf.nn.bias_add(conv,b),name="h") # bz, n, 1, dc

        # conv = tf.multiply(tf.reshape(h, [bz, n, dc]), tf.reshape(alpha, [bz, n, 1])) # bz, n, dc
        # h = tf.reshape(conv, [bz, n, 1, dc])

        # U: [dc, dc]
        R = tf.reshape(h, [bz, n, dc])
        U = tf.get_variable(initializer=initializer,shape=[dc,dc],name='U')
        G = tf.matmul(tf.reshape(R, [bz*n, dc]), U)
        G = tf.matmul(G, tf.transpose(rel_embed)) 
        G = tf.reshape(G, [bz, n, nr])
        AP = tf.nn.softmax(G, dim=1)# attention pooling tensor

        wo = tf.matmul(tf.transpose(R, perm=[0, 2, 1]),AP)# (bz, dc, nr)
        wo = tf.reduce_max(wo, axis=-1) # (bz, dc)
        pooled_outputs.append(wo)

        # # max pooling
        # pooled = tf.nn.max_pool(h,
        #                     ksize=[1,n,1,1],
        #                     strides=[1,n,1,1],
        #                     padding="SAME"
        #       )
        # pooled_outputs.append(pooled)
    # h_pool = tf.concat(pooled_outputs, 3)
    # h_pool_flat = tf.reshape(h_pool,[-1,dc*len(filter_sizes)])
    h_pool_flat = tf.concat(pooled_outputs, -1)

    # e embdding
    e_flat = tf.concat([e1, e2], 2)
    e_flat = tf.reshape(e_flat,[-1,dw*6])
    all_flat = tf.concat([h_pool_flat, e_flat],1)
    # h_pool_flat = all_flat
    h_pool_flat = tf.reshape(all_flat,[-1,dc*len(filter_sizes) + dw * 6])


    if is_training and keep_prob < 1:
      h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob)
    
    # output
    W_o = tf.get_variable(initializer=initializer,shape=[dc*len(filter_sizes) + dw * 6,nr],name='w_o')
    b_o = tf.get_variable(initializer=initializer,shape=[nr],name='b_o')
    scores = tf.nn.xw_plus_b(h_pool_flat,W_o,b_o,name="scores")
    predict = tf.argmax(scores,1,name="predictions")
    predict = tf.cast(predict, dtype=tf.int32)
    acc = tf.reduce_sum(tf.cast(tf.equal(predict, in_y), dtype=tf.int32))
    self.predict = predict
    self.acc = acc

    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=scores, 
                                              labels=tf.one_hot(in_y, nr))
    )
    l2_loss = tf.nn.l2_loss(W_o)
    l2_loss += tf.nn.l2_loss(b_o)
    l2_loss = config.l2_reg_lambda * l2_loss
    
    self.loss = loss + l2_loss

    if not is_training:
      return

    # optimizer 
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
    self.reg_op = tf.no_op()
    self.global_step = global_step

    
