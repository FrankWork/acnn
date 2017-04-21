import tensorflow as tf
import numpy as np
import logging
import os

import config
import utils

config = config.FLAGS

class Model(object):
  def __init__(self, embeddings, is_training=True):
    bz = config.batch_size
    ez = config.embedding_size

    in_x = tf.placeholder(dtype=tf.int32, shape=[bz,None], name='x')
    in_e1 = tf.placeholder(dtype=tf.int32, shape=[bz,None], name='e1')
    in_e2 = tf.placeholder(dtype=tf.int32, shape=[bz,None], name='e2')
    in_dist1 = tf.placeholder(dtype=tf.int32, shape=[bz,None], name='dist1')
    in_dist2 = tf.placeholder(dtype=tf.int32, shape=[bz,None], name='dist2')
    in_y = tf.placeholder(dtype=tf.int32, shape=[bz,None], name='dist2')
    
    embed = tf.get_variable(initializer=embeddings, dtype=tf.float32, name='embed')
    pos_embed = tf.get_variable(initializer=tf.truncated_normal_initializer(),
                              shape=[],dtype=tf.float32,name='pos_embed')
    
    x = tf.nn.embedding_lookup(embed, in_x, name='x') # bz,len,ez
    e1 = tf.nn.embedding_lookup(embed, in_e1, name='e1')#
    e2 = tf.nn.embedding_lookup(embed, in_e2, name='e2')
    dist1 = tf.nn.embedding_lookup(pos_embed, in_dist1, name='dist1')#bz, len, pz
    dist2 = tf.nn.embedding_lookup(pos_embed, in_dist2, name='dist2')# bz, len, pz

    x = tf.concat([x, dist1, dist2], 2) # bz, len, ez+2*pz




    (sents_vec, relations, e1_pos, e2_pos, dist1, dist2)


def init():
  path = config.data_path
  config.embedding_file = os.path.join(path, config.embedding_file)
  config.embedding_vocab = os.path.join(path, config.embedding_vocab)
  config.train_file = os.path.join(path, config.train_file)
  config.test_file = os.path.join(path, config.test_file)

  # Config log
  if config.log_file is None:
    logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
  else:
    logging.basicConfig(filename=config.log_file,
                      filemode='w', level=logging.DEBUG,
                      format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
  # Load data
  # data = (sentences, relations, e1_pos, e2_pos)
  train_data = utils.load_data(config.train_file)
  test_data = utils.load_data(config.test_file)

  logging.info('trian data: %d' % len(train_data[0]))
  logging.info('test data: %d' % len(test_data[0]))

  # Build vocab
  word_dict = utils.build_dict(train_data[0] + test_data[0])
  logging.info('total words: %d' % len(word_dict))

  embeddings = utils.load_embedding(config.embedding_file, config.embedding_vocab,
                                    word_dict)

  # Log parameters
  flags = config.__dict__['__flags']
  flag_str = "\n"
  for k in flags:
    flag_str += "\t%s:\t%s\n" % (k, flags[k])
  logging.info(flag_str)

  # vectorize data
  # vec = (sents_vec, relations, e1_vec, e2_vec, dist1, dist2)
  train_vec = utils.vectorize(train_data, word_dict)
  test_vec = utils.vectorize(test_data, word_dict)
  
  return embeddings, train_vec, test_vec

  
def main(_):
  embeddings, train_vec, test_vec = init()
  sents_vec, relations, e1_vec, e2_vec, dist1, dist2 = test_vec
  



if __name__ == '__main__':
  tf.app.run()
