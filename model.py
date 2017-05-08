import tensorflow as tf

# step 2000, 0.69
# Epoch: 100 Train: 82.05% Test: 63.78%
class Model(object):
  def __init__(self, config, embeddings, is_training=True):
    bz = config.batch_size
    dw = config.embedding_size
    dp = config.pos_embed_size
    np = config.pos_embed_num
    n = config.max_len
    # k = config.slide_window
    dc = config.num_filters
    nr = config.classnum # number of relations
    keep_prob = config.keep_prob

    # input
    in_x = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_x') # sentences
    in_e1 = tf.placeholder(dtype=tf.int32, shape=[bz, 3], name='in_e1')
    in_e2 = tf.placeholder(dtype=tf.int32, shape=[bz, 3], name='in_e2')
    in_dist1 = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_dist1')
    in_dist2 = tf.placeholder(dtype=tf.int32, shape=[bz,n], name='in_dist2')
    in_y = tf.placeholder(dtype=tf.int32, shape=[bz], name='in_y') # relations
    
    self.inputs = (in_x, in_e1, in_e2, in_dist1, in_dist2, in_y)
    
    # embeddings
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    embed = tf.get_variable(initializer=embeddings, dtype=tf.float32, name='word_embed')
    pos1_embed = tf.get_variable(initializer=initializer,shape=[np, dp],name='position1_embed')
    pos2_embed = tf.get_variable(initializer=initializer,shape=[np, dp],name='position2_embed')
    # rel_embed = tf.get_variable(initializer=initializer,shape=[nr, dc],name='relation_embed')

    # embdding lookup
    e1 = tf.nn.embedding_lookup(embed, in_e1, name='e1')# bz,dw
    e2 = tf.nn.embedding_lookup(embed, in_e2, name='e2')# bz,dw
    x = tf.nn.embedding_lookup(embed, in_x, name='x')   # bz,n,dw
    dist1 = tf.nn.embedding_lookup(pos1_embed, in_dist1, name='dist1')#bz, n, k,dp
    dist2 = tf.nn.embedding_lookup(pos2_embed, in_dist2, name='dist2')# bz, n, k,dp
    # y = tf.nn.embedding_lookup(rel_embed, in_y, name='y')# bz, dc

    # convolution
    # x: (batch_size, max_len, embdding_size, 1)
    # w: (filter_size, embdding_size, 1, num_filters)
    d = dw+2*dp
    filter_sizes = [3, 4, 5]
    pooled_outputs = []
    x_conv = tf.reshape(tf.concat([x, dist1, dist2], -1), # bz, n, d
                            [bz,n,d,1])
    if is_training and keep_prob < 1:
      x_conv = tf.nn.dropout(x_conv, keep_prob)

    for i, k in enumerate(filter_sizes):
      with tf.variable_scope("conv-%d" % k):# , reuse=False
        w = tf.get_variable(initializer=initializer,shape=[k, d, 1, dc],name='weight')
        b = tf.get_variable(initializer=initializer,shape=[dc],name='bias')
        conv = tf.nn.conv2d(x_conv, w, strides=[1,1,d,1],padding="SAME")

        h = tf.nn.tanh(tf.nn.bias_add(conv,b),name="h") # bz, n, 1, dc

        # max pooling
        pooled = tf.nn.max_pool(h,
                            ksize=[1,n,1,1],
                            strides=[1,n,1,1],
                            padding="SAME"
              )
        pooled_outputs.append(pooled)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool,[-1,dc*len(filter_sizes)])

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

    
