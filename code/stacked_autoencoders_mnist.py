# https://github.com/ageron/handson-ml/blob/master/15_autoencoders.ipynb

import logging
import os
from functools import partial
from time import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap='Greys', interpolation='nearest')
    plt.axis('off')


def show_reconstructed_digits(X, outputs, model_path=None, n_test_digits=2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])


def save_fig(fig_id, arg_folder, tight_layout=True):
    path = os.path.join(arg_folder, fig_id + '.png')
    print('Saving figure', fig_id)
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

    mnist = input_data.read_data_sets(train_dir='../data/mnist/')

    n_inputs = 28 * 28
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

    n_epochs = 5
    batch_size = 150

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = mnist.train.num_examples // batch_size
            for iteration in range(n_batches):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
            logger.info('epoch: %s train MSE %.4f' % (epoch, loss_train))
            saver.save(sess, './models/my_model_all_layers.ckpt')

    show_reconstructed_digits(X, outputs, './models/my_model_all_layers.ckpt')
    save_fig('reconstruction_plot', '../output/mnist/')
