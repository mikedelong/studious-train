import logging
import os
from functools import partial
from pickle import load
from time import time

import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import choice


def plot_image(image, shape=None):
    plt.imshow(image.reshape(shape), cmap='Greys', interpolation='nearest')
    plt.axis('off')


def show_reconstructed(x, arg_output, model_path=None, n_test_samples=2):
    with tf.Session() as local_session:
        if model_path:
            saver.restore(local_session, model_path)
        x_test = circles_data[:n_test_samples]
        outputs_val = arg_output.eval(feed_dict={x: x_test})

    _ = plt.figure(figsize=(8, 3 * n_test_samples))
    for index in range(n_test_samples):
        plt.subplot(n_test_samples, 2, index * 2 + 1)
        plot_image(x_test[index], [40, 40])
        plt.subplot(n_test_samples, 2, index * 2 + 2)
        plot_image(outputs_val[index], [40, 40])


def save_fig(fig_id, arg_folder, arg_logger, tight_layout=True):
    suffix = '.png'
    path = os.path.join(arg_folder, fig_id + suffix)
    arg_logger.info('Saving figure %s' % fig_id + suffix)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


if __name__ == '__main__':
    start_time = time()

    console_formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.info('started')

    circles_file = '../data/circles.pkl'
    with open(circles_file, 'rb') as circles_fp:
        circles_data = load(circles_fp)
    logger.info('loaded %d items from %s' % (len(circles_data), circles_file))

    n_inputs = 40 * 40
    n_hidden1 = 300
    n_hidden2 = 150
    n_hidden3 = n_hidden1
    n_outputs = n_inputs

    learning_rate = 0.01
    l2_reg = 0.0001

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    he_init = tf.contrib.layers.variance_scaling_initializer()
    l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
    my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=he_init,
                             kernel_regularizer=l2_regularizer)
    hidden1 = my_dense_layer(X, n_hidden1)
    hidden2 = my_dense_layer(hidden1, n_hidden2)
    hidden3 = my_dense_layer(hidden2, n_hidden3)
    outputs = my_dense_layer(hidden3, n_outputs, activation=None)

    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([reconstruction_loss] + reg_losses)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 10
    batch_size = 15
    num_examples = len(circles_data)

    with tf.Session() as session:
        init.run()
        for epoch in range(n_epochs):
            n_batches = num_examples // batch_size
            X_batch = None
            for iteration in range(n_batches):
                X_batch = circles_data[choice(num_examples, batch_size, replace=False), :]
                session.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
            logger.info('epoch: %s train MSE %.4f' % (epoch, loss_train))
            saver.save(session, './models/circle_model_all_layers.ckpt')

    show_reconstructed(X, outputs, './models/circle_model_all_layers.ckpt')
    save_fig('reconstruction_plot', '../output/circles/', arg_logger=logger, tight_layout=True)
