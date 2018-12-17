import logging
import os
from json import load as load_json
from pickle import load as load_pickle
from time import time

import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import choice


def plot_image(image, shape=None):
    plt.imshow(image.reshape(shape), cmap='Greys', interpolation='nearest')
    plt.axis('off')


def show_reconstructed(x, arg_output, model_path, n_test_samples, arg_shape):
    with tf.Session() as local_session:
        if model_path:
            saver.restore(local_session, model_path)
        x_test = shapes_data[:n_test_samples]
        outputs_val = arg_output.eval(feed_dict={x: x_test})

    _ = plt.figure(figsize=(8, 3 * n_test_samples))
    for index in range(n_test_samples):
        plt.subplot(n_test_samples, 2, index * 2 + 1)
        plot_image(x_test[index], arg_shape)
        plt.subplot(n_test_samples, 2, index * 2 + 2)
        plot_image(outputs_val[index], arg_shape)


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

    with open('autoencoder_settings.json', 'r') as settings_fp:
        settings = load_json(settings_fp)

        shapes_file = settings['shapes_file']
        image_height = settings['image_height']
        image_width = settings['image_width']
        image_shape = [image_height, image_width]
        model_checkpoint = settings['model_checkpoint']
        n_epochs = settings['n_epochs']
        batch_size = settings['batch_size']
        test_size = settings['test_size']

    with open(shapes_file, 'rb') as shapes_fp:
        shapes_data = load_pickle(shapes_fp)
    logger.info('loaded %d items from %s' % (len(shapes_data), shapes_file))

    n_inputs = image_height * image_width

    n_hidden1 = 300
    n_hidden2 = 150
    n_hidden3 = n_hidden1
    n_outputs = n_inputs

    learning_rate = 0.01
    l2_reg = 0.0001

    activation = tf.nn.elu
    regularlizer = tf.contrib.layers.l2_regularizer(l2_reg)
    initializer = tf.contrib.layers.variance_scaling_initializer()

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    weights1_init = initializer([n_inputs, n_hidden1])
    weights2_init = initializer([n_hidden1, n_hidden2])
    weights1 = tf.Variable(weights1_init, dtype=tf.float32, name='weights1')
    weights2 = tf.Variable(weights2_init, dtype=tf.float32, name='weights2')
    weights3 = tf.transpose(weights2, name='weights3')
    weights4 = tf.transpose(weights1, name='weights4')

    biases1 = tf.Variable(tf.zeros(n_hidden1), name='biases1')
    biases2 = tf.Variable(tf.zeros(n_hidden2), name='biases2')
    biases3 = tf.Variable(tf.zeros(n_hidden3), name='biases3')
    biases4 = tf.Variable(tf.zeros(n_outputs), name='biases4')

    hidden1 = activation(tf.matmul(X, weights1) + biases1)
    hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
    hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
    outputs = tf.matmul(hidden3, weights4) + biases4

    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
    reg_losses = regularlizer(weights1) + regularlizer(weights2)
    loss = reconstruction_loss + reg_losses

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    num_examples = len(shapes_data)

    # do the test/train split
    train_size = num_examples - test_size
    with tf.Session() as session:
        init.run()
        for epoch in range(n_epochs):
            n_batches = train_size // batch_size
            X_batch = None
            for iteration in range(n_batches):
                X_batch = shapes_data[choice(train_size, batch_size, replace=False), :]
                session.run(training_op, feed_dict={X: X_batch})
            loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
            logger.info('epoch: %s train MSE %.4f' % (epoch, loss_train))
            saver.save(session, model_checkpoint)

    show_reconstructed(X, outputs, model_checkpoint, n_test_samples=test_size, arg_shape=image_shape)
    save_fig('reconstruction_plot', '../output/circles/tied/', arg_logger=logger, tight_layout=True)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
