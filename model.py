import tensorflow as tf


class Model(object):
  def __init__(self, config, embeddings, is_training=True):
    bz = config.batch_size
    dw = config.embedding_size
    dp = config.pos_embed_size
    np = config.pos_embed_num
    n = config.max_len
    k = config.slide_window
    dc = config.num_filters
    nr = config.classnum # number of relations
    keep_prob = config.keep_prob

    # input
    in_x = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_x') # sentences
    in_e1 = tf.placeholder(dtype=tf.int32, shape=[bz], name='in_e1')
    in_e2 = tf.placeholder(dtype=tf.int32, shape=[bz], name='in_e2')
    in_dist1 = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_dist1')
    in_dist2 = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_dist2')
    in_y = tf.placeholder(dtype=tf.int32, shape=[bz], name='in_y') # relations
    
    self.inputs = (in_x, in_e1, in_e2, in_dist1, in_dist2, in_y)
    
    # embeddings
    initializer = tf.truncated_normal_initializer()
    embed = tf.get_variable(initializer=embeddings, dtype=tf.float32, name='word_embed')
    pos1_embed = tf.get_variable(initializer=initializer,shape=[np, dp],name='position1_embed')
    pos2_embed = tf.get_variable(initializer=initializer,shape=[np, dp],name='position2_embed')
    rel_embed = tf.get_variable(initializer=initializer,shape=[nr, dc],name='relation_embed')

    # embdding lookup
    e1 = tf.nn.embedding_lookup(embed, in_e1, name='e1')# bz,dw
    e2 = tf.nn.embedding_lookup(embed, in_e2, name='e2')# bz,dw
    x = tf.nn.embedding_lookup(embed, in_x, name='x')   # bz,n,dw
    dist1 = tf.nn.embedding_lookup(pos1_embed, in_dist1, name='dist1')#bz, n, k,dp
    dist2 = tf.nn.embedding_lookup(pos2_embed, in_dist2, name='dist2')# bz, n, k,dp
    y = tf.nn.embedding_lookup(rel_embed, in_y, name='y')# bz, dc


    # input attention
    def inner_product(e, x):
      '''
      <x, y> = x1y1 + x2y2 + ... + xnyn
      e:        (bz, dw) => (bz, 1, dw)
      x:        (bz, n, dw)
      return :  (bz, n)
      '''
      return tf.reduce_sum(
                tf.multiply(tf.reshape(e, [bz, 1, dw]), x), 
                -1
              )

    alpha1 = tf.nn.softmax(inner_product(e1, x))# bz, n
    alpha2 = tf.nn.softmax(inner_product(e2, x))# bz, n
    alpha = (alpha1 + alpha2)/2

    # convolution
    # x: (batch_size, max_len, embdding_size, 1)
    # w: (filter_size, embdding_size, 1, num_filters)
    d = dw+2*dp
    x_conv = tf.reshape(tf.concat([x, dist1, dist2], -1), # bz, n, d
                        [bz,n,d,1])
    if is_training and keep_prob < 1:
      x_conv = tf.nn.dropout(x_conv, keep_prob)

    w = tf.get_variable(initializer=initializer,shape=[k, d, 1, dc],name='weight')
    b = tf.get_variable(initializer=initializer,shape=[dc],name='bias')
    conv = tf.nn.conv2d(x_conv, w, strides=[1,1,d,1],padding="SAME")
    r = tf.multiply(tf.reshape(conv, [bz, n, dc]), tf.reshape(alpha, [bz, n, 1])) # bz, n, 1, dc

    R = tf.nn.tanh(tf.nn.bias_add(r,b),name="R") # bz, n, 1, dc
    R = tf.reshape(R, [bz, n, dc])

    # attention pooling
    U = tf.get_variable(initializer=initializer,shape=[dc,dc],name='U')
    
    # batch matmul
    # G = R * U * WL
    # R: (bz, n, dc)
    # U: (dc, dc)
    # WL:(dc, nr)
    G = tf.matmul(# (bz*n,dc), (dc, dc) => (bz*n, dc)
      tf.reshape(R, [bz*n, dc]), U
    )
    G = tf.matmul(# (bz*n, dc), (dc, nr) => (bz*n, nr)
      G, tf.transpose(rel_embed)
    ) 
    G = tf.reshape(G, [bz, n, nr])
    AP = tf.nn.softmax(G, dim=1)# attention pooling tensor

    # predict
    wo = tf.matmul(
      tf.transpose(R, perm=[0, 2, 1]), # batch transpose: (bz, n, dc) => (bz,dc,n)
      AP
    )# (bz, dc, nr)
    wo = tf.reduce_max(wo, axis=-1) # (bz, dc)
    # wo = tf.nn.max_pool(tf.expand_dims(wo,-1),
    #                     ksize=[1,1,nr,1],
    #                     strides=[1,1,nr,1],
    #                     padding="SAME"
    #       )# (bz, dc, 1, 1)
    # wo=tf.reshape(wo,[bz, dc])

    if is_training and keep_prob < 1:
      wo = tf.nn.dropout(wo, keep_prob)

    # accuracy
    def distance(wo, y, axis=None):
      return tf.norm(
                tf.nn.l2_normalize(wo, dim=-1) - y,
                axis = axis
            )# a scalar value


    all_dist = distance(
              tf.tile(tf.expand_dims(wo, axis=1), [1, nr, 1]), # bz, nr, dc
              rel_embed,# nr, dc
              axis=-1
    )# bz, nr
    predict = tf.argmin(all_dist, axis=-1)
    predict = tf.cast(predict, dtype=tf.int32)
    acc = tf.reduce_sum(tf.cast(tf.equal(predict, in_y), dtype=tf.int32))
    self.predict = predict
    self.acc = acc

    if not is_training:
      return

    # train
    mask = tf.one_hot(in_y, nr, on_value=1000., off_value=0.)# bz, nr
    neg_dist = tf.add(all_dist, mask)
    neg_y = tf.argmin(neg_dist, axis=-1)# bz, 1
    neg_y = tf.nn.embedding_lookup(rel_embed, neg_y)# bz, dc

    l2_loss = tf.nn.l2_loss(rel_embed)
    l2_loss += tf.nn.l2_loss(U)
    l2_loss += tf.nn.l2_loss(w)
    l2_loss += tf.nn.l2_loss(b)
    l2_loss = config.l2_reg_lambda * l2_loss

    loss = distance(wo, y) + (config.margin - distance(wo, neg_y))
    self.loss = loss

    # optimizer 
    # optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    optimizer2 = tf.train.AdamOptimizer(config.learning_rate2)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      config.grad_clipping)
    capped_gvs = zip(grads, tvars)

    tf.logging.set_verbosity(tf.logging.ERROR)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    reg_op = optimizer2.minimize(l2_loss)



    self.train_op = train_op
    self.reg_op = reg_op
    # self.reg_op = tf.no_op()
    self.global_step = global_step

    # max pooling
    # pooled = tf.nn.max_pool(tf.expand_dims(R, axis=-1),
    #                     ksize=[1,n,1,1],
    #                     strides=[1,n,1,1],
    #                     padding="SAME"
    #       )
    # h_pool_flat = tf.reshape(pooled,[-1,dc])
    # if is_training and keep_prob < 1:
    #   h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob)
    # W_o = tf.get_variable(initializer=initializer,shape=[dc,nr],name='w_o')
    # b_o = tf.get_variable(initializer=initializer,shape=[nr],name='b_o')
    # scores = tf.nn.xw_plus_b(h_pool_flat,W_o,b_o,name="scores")
    # predict = tf.argmax(scores,1,name="predictions")
    # predict = tf.cast(predict, dtype=tf.int32)
    # acc = tf.reduce_sum(tf.cast(tf.equal(predict, in_y), dtype=tf.int32))
    # self.predict = predict
    # self.acc = acc
    # loss = tf.reduce_mean(
    #   tf.nn.softmax_cross_entropy_with_logits(logits=scores, 
    #                                           labels=tf.one_hot(in_y, nr))
    # )
    # l2_loss = tf.nn.l2_loss(W_o)
    # l2_loss += tf.nn.l2_loss(b_o)
    # l2_loss = config.l2_reg_lambda * l2_loss
