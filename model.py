import tensorflow as tf

# 05-08 23:35 Epoch: 1 Train: 11.53% Test: 30.11%
# 05-08 23:35 Epoch: 2 Train: 23.43% Test: 43.70%
# 05-08 23:35 Epoch: 3 Train: 33.38% Test: 49.48%
# 05-08 23:35 Epoch: 4 Train: 40.21% Test: 54.37%
# 05-08 23:35 Epoch: 5 Train: 43.60% Test: 56.85%
# 05-08 23:35 Epoch: 6 Train: 47.60% Test: 59.00%
# 05-08 23:36 Epoch: 7 Train: 50.79% Test: 59.59%
# 05-08 23:36 Epoch: 8 Train: 54.76% Test: 61.81%
# 05-08 23:36 Epoch: 9 Train: 56.53% Test: 63.07%
# 05-08 23:36 Epoch: 10 Train: 59.44% Test: 65.15%
# 05-08 23:36 Epoch: 11 Train: 62.09% Test: 66.52%
# 05-08 23:36 Epoch: 12 Train: 63.19% Test: 67.19%
# 05-08 23:36 Epoch: 13 Train: 65.16% Test: 69.11%
# 05-08 23:37 Epoch: 14 Train: 67.06% Test: 69.56%
# 05-08 23:37 Epoch: 15 Train: 68.46% Test: 70.15%
# 05-08 23:37 Epoch: 16 Train: 69.14% Test: 70.81%
# 05-08 23:37 Epoch: 17 Train: 69.97% Test: 70.85%
# 05-08 23:37 Epoch: 18 Train: 71.65% Test: 70.41%
# 05-08 23:37 Epoch: 19 Train: 72.67% Test: 71.00%
# 05-08 23:38 Epoch: 20 Train: 73.04% Test: 71.96%
# 05-08 23:38 Epoch: 21 Train: 74.29% Test: 71.89%
# 05-08 23:38 Epoch: 22 Train: 74.52% Test: 72.00%
# 05-08 23:38 Epoch: 23 Train: 76.53% Test: 72.37%
# 05-08 23:38 Epoch: 24 Train: 76.92% Test: 72.81%
# 05-08 23:38 Epoch: 25 Train: 76.89% Test: 73.11%
# 05-08 23:38 Epoch: 26 Train: 78.85% Test: 73.07%
# 05-08 23:39 Epoch: 27 Train: 79.89% Test: 73.22%
# 05-08 23:39 Epoch: 28 Train: 79.83% Test: 73.59%
# 05-08 23:39 Epoch: 29 Train: 80.04% Test: 74.04%
# 05-08 23:39 Epoch: 30 Train: 81.19% Test: 72.96%
# 05-08 23:39 Epoch: 31 Train: 82.05% Test: 73.52%
# 05-08 23:39 Epoch: 32 Train: 82.09% Test: 73.93%
# 05-08 23:39 Epoch: 33 Train: 82.78% Test: 74.19%
# 05-08 23:40 Epoch: 34 Train: 83.35% Test: 74.41%
# 05-08 23:40 Epoch: 35 Train: 83.23% Test: 73.70%
# 05-08 23:40 Epoch: 36 Train: 83.90% Test: 74.22%
# 05-08 23:40 Epoch: 37 Train: 84.28% Test: 73.96%
# 05-08 23:40 Epoch: 38 Train: 84.69% Test: 74.30%
# 05-08 23:40 Epoch: 39 Train: 85.88% Test: 74.78%
# 05-08 23:41 Epoch: 40 Train: 86.20% Test: 75.11%
# 05-08 23:41 Epoch: 41 Train: 86.66% Test: 74.37%
# 05-08 23:41 Epoch: 42 Train: 87.15% Test: 74.85%
# 05-08 23:41 Epoch: 43 Train: 87.64% Test: 75.15%
# 05-08 23:41 Epoch: 44 Train: 87.76% Test: 75.41%
# 05-08 23:41 Epoch: 45 Train: 87.85% Test: 75.59%
# 05-08 23:41 Epoch: 46 Train: 89.20% Test: 75.37%
# 05-08 23:42 Epoch: 47 Train: 88.59% Test: 75.81%
# 05-08 23:42 Epoch: 48 Train: 88.66% Test: 76.07%
# 05-08 23:42 Epoch: 49 Train: 89.80% Test: 75.52%
# 05-08 23:42 Epoch: 50 Train: 89.76% Test: 75.81%

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

    
