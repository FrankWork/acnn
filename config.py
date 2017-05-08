import tensorflow as tf


# Basics
# tf.app.flags.DEFINE_boolean("debug", True,
#                             "run in the debug mode.")
tf.app.flags.DEFINE_boolean("test_only", False,
                            "no need to run training process.")

# Data files
tf.app.flags.DEFINE_string("data_path", "data/", "Data directory")
tf.app.flags.DEFINE_string("embedding_file", "embedding/senna/embeddings.txt", 
                           "embedding file")
tf.app.flags.DEFINE_string("embedding_vocab", "embedding/senna/words.lst", 
                           "embedding vocab file")
tf.app.flags.DEFINE_string("train_file", "train.txt", "training file")
tf.app.flags.DEFINE_string("test_file", "test.txt", "Test file")
tf.app.flags.DEFINE_string("log_file", None, "Log file")
tf.app.flags.DEFINE_string("save_path", None, "save model here")
# tf.app.flags.DEFINE_string("log_file", 'run/log.txt', "Log file")
# tf.app.flags.DEFINE_string("save_path", 'run/', "save model here")
tf.app.flags.DEFINE_string("pad_word", "<PAD>", "save model here")


# Model details
tf.app.flags.DEFINE_integer("pos_embed_num", 123, "position embedding number")
tf.app.flags.DEFINE_integer("pos_embed_size", 5, "position embedding size")
tf.app.flags.DEFINE_integer("slide_window", 3, "Slide window size")
tf.app.flags.DEFINE_integer("num_filters", 100, 
                            "How many features a convolution op have to output")
tf.app.flags.DEFINE_integer("classnum", 19, "Number of relations")

# Optimization details
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size")
tf.app.flags.DEFINE_integer("num_epoches", 100, "Number of epoches")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "Dropout keep prob.")
tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate.")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0, "regularization parameter")
tf.app.flags.DEFINE_float("learning_rate2", 1e-3, "learning_rate for regularization")
tf.app.flags.DEFINE_float("margin", 1, "margin based loss function")
tf.app.flags.DEFINE_float("grad_clipping", 10., "Gradient clipping.")


FLAGS = tf.app.flags.FLAGS