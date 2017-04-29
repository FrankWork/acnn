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
    pos_embed = tf.get_variable(initializer=initializer,shape=[np, dp],name='position_embed')
    rel_embed = tf.get_variable(initializer=initializer,shape=[nr, dc],name='relation_embed')

    # slide window
    hk = (k-1)//2 # half of the slide window size
    def slide_window(x):
      sw_x = tf.pad(x, [[0,0], [hk,hk]], "CONSTANT")# bz, n+2*hk
      sw_x = tf.map_fn(lambda i: sw_x[:, i-hk:i+hk+1], tf.range(hk, n+hk), dtype=tf.int32)
      return tf.stack(tf.unstack(sw_x), axis=1)
    
    sw_x = slide_window(in_x)         # bz, n, k
    sw_dist1 = slide_window(in_dist1) # bz, n, k
    sw_dist2 = slide_window(in_dist2) # bz, n, k

    # embdding lookup
    e1 = tf.nn.embedding_lookup(embed, in_e1, name='e1')# bz,dw
    e2 = tf.nn.embedding_lookup(embed, in_e2, name='e2')# bz,dw
    x = tf.nn.embedding_lookup(embed, in_x, name='x')   # bz,n,dw
    x_k = tf.nn.embedding_lookup(embed, sw_x, name='x_k') # bz,n,k,dw
    dist1 = tf.nn.embedding_lookup(pos_embed, sw_dist1, name='dist1')#bz, n, k,dp
    dist2 = tf.nn.embedding_lookup(pos_embed, sw_dist2, name='dist2')# bz, n, k,dp
    y = tf.nn.embedding_lookup(rel_embed, in_y, name='y')# bz, dc

    z = tf.concat([x_k, dist1, dist2], -1) # bz, n, k, dw+2*dp
    z = tf.reshape(z, [bz, n, k*(dw+2*dp)]) # bz, n, k*(dw+2*dp)
    if is_training and keep_prob < 1:
      z = tf.nn.dropout(z, keep_prob)

    # input attention
    def inner_product(e, x):
      '''
      <x, y> = x1y1 + x2y2 + ... + xnyn
      e:        (bz, dw) => (bz, 1, dw)
      x:        (bz, n, dw)
      return :  (bz, n)
      '''
      return tf.reduce_sum(
                tf.multiply(tf.reshape(e1, [bz, 1, dw]), x), 
                -1
             )

    alpha1 = tf.nn.softmax(inner_product(e1, x))# bz, n
    alpha2 = tf.nn.softmax(inner_product(e2, x))# bz, n
    alpha = (alpha1 + alpha2)/2

    r = tf.multiply(z, tf.reshape(alpha, [bz, n, 1])) # bz, n, k*d,   d=(dw+2*dp)

    # convolution
    # x: (batch_size, max_len, embdding_size, 1)
    # w: (filter_size, embdding_size, 1, num_filters)
    d = dw+2*dp
    w = tf.get_variable(initializer=initializer,shape=[1, k*d, 1, dc],name='weight')
    b = tf.get_variable(initializer=initializer,shape=[dc],name='bias')
    conv = tf.nn.conv2d(tf.reshape(r, [bz,n,k*d,1]), w, strides=[1,1,k*d,1],padding="SAME")
    R = tf.nn.tanh(tf.nn.bias_add(conv,b),name="R") # bz, n, 1, dc
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
    AP = tf.nn.softmax(G, dim=0)# attention pooling tensor

    # predict
    wo = tf.matmul(
      tf.transpose(R, perm=[0, 2, 1]), # batch transpose: (bz, n, dc) => (bz,dc,n)
      AP
    )# (bz, dc, nr)
    wo = tf.reduce_max(wo, axis=-1) # (bz, dc)

    if is_training and keep_prob < 1:
      wo = tf.nn.dropout(wo, keep_prob)
    
    def distance(wo, y, axis=None):
      return tf.norm(
        tf.nn.l2_normalize(wo, dim=-1) - y,
        axis = axis
      )# a scalar value
    
    # accuracy
    all_dist = distance(
              tf.tile(tf.expand_dims(wo, axis=1), [1, nr, 1]), # bz, nr, dc
              rel_embed,# bz, dc
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
    mask = tf.one_hot(in_y, nr, on_value=1000., off_value=1.)# bz, nr
    neg_dist = tf.multiply(all_dist, mask)
    neg_y = tf.argmin(neg_dist, axis=-1)# bz, 1
    neg_y = tf.nn.embedding_lookup(rel_embed, neg_y)# bz, dc

    l2_loss = tf.nn.l2_loss(rel_embed)
    l2_loss += tf.nn.l2_loss(U)
    l2_loss += tf.nn.l2_loss(w)
    l2_loss += tf.nn.l2_loss(b)

    loss = distance(wo, y) + (1-distance(wo, neg_y)) + config.l2_reg_lambda * l2_loss
    self.loss = loss

    # optimizer 
    optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      config.grad_clipping)
    capped_gvs = zip(grads, tvars)

    # tf.logging.set_verbosity(tf.logging.WARN)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    self.train_op = train_op
    self.global_step = global_step



