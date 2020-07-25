from __future__ import division
from __future__ import print_function

from gcn.utils import *
from gcn.inits import *
from gcn.models import GCN, MLP
import scipy.sparse as sp
import tensorflow as tf
import time


def ini_net_settings(dataset="default", model='gcn', epochs=1, learning_rate=0.01, hidden1=16, dropout=0.5,
                     layer_num=3):
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', dataset, 'Dataset name.')
    flags.DEFINE_integer('layer_num', layer_num, 'layer number.')
    flags.DEFINE_string('model', model, 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', epochs, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', hidden1, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', dropout, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
    return FLAGS


def get_embeddings(adj, features, output_shape, net_settings):
    FLAGS = net_settings
    y_train, train_mask = ini_labels(output_shape)

    # Some pre-processing
    fea = sp.csr_matrix(features).tolil()
    features = preprocess_features(fea)

    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []
    outputs = None

    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        results = sess.run([model.opt_op, model.loss, model.outputs], feed_dict=feed_dict)
        cost_val.append(results[1])
        outputs = results[2]

        # # Print results
        # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(results[1]),
        #       "train_acc=", "{:.5f}".format(results[2]), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")
    return outputs
