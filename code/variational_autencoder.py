import logging
import os
from functools import partial
from json import load as load_json
from pickle import load as load_pickle
from time import time

import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import choice
from numpy.random import normal


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

    with open('variational_autoencoder.json', 'r') as settings_fp:
        settings = load_json(settings_fp)
        logger.info(settings)

        batch_size = settings['batch_size']
        image_height = settings['image_height']
        image_width = settings['image_width']
        model_checkpoint = settings['model_checkpoint']
        n_epochs = settings['n_epochs']
        output_folder = settings['output_folder']
        shapes_file = settings['shapes_file']
        test_size = settings['test_size']
        image_shape = [image_height, image_width]

    with open(shapes_file, 'rb') as shapes_fp:
        shapes_data = load_pickle(shapes_fp)
    logger.info('loaded %d items from %s' % (len(shapes_data), shapes_file))

    n_inputs = image_height * image_width

    n_hidden1 = 500
    n_hidden2 = 500
    n_hidden3 = 20
    n_hidden4 = n_hidden2
    n_hidden5 = n_hidden1
    n_outputs = n_inputs
    learning_rate = 0.001

    initializer = tf.contrib.layers.variance_scaling_initializer()
    my_dense_layer = partial(
        tf.layers.dense,
        activation=tf.nn.elu,
        kernel_initializer=initializer)

    X = tf.placeholder(tf.float32, [None, n_inputs])
    hidden1 = my_dense_layer(X, n_hidden1)
    hidden2 = my_dense_layer(hidden1, n_hidden2)
    hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation=None)
    hidden3_gamma = my_dense_layer(hidden2, n_hidden3, activation=None)
    noise = tf.random_normal(tf.shape(hidden3_gamma), dtype=tf.float32)
    hidden3 = hidden3_mean + tf.exp(0.5 * hidden3_gamma) * noise
    hidden4 = my_dense_layer(hidden3, n_hidden4)
    hidden5 = my_dense_layer(hidden4, n_hidden5)
    logits = my_dense_layer(hidden5, n_outputs, activation=None)
    outputs = tf.sigmoid(logits)

    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
    reconstruction_loss = tf.reduce_sum(xentropy)
    latent_loss = 0.5 * tf.reduce_sum(
        tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1.0 - hidden3_gamma)
    loss = reconstruction_loss + latent_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    num_examples = len(shapes_data)

    train_size = num_examples
    with tf.Session() as session:
        init.run()
        for epoch in range(n_epochs):
            n_batches = train_size // batch_size
            X_batch = None
            for iteration in range(n_batches):
                X_batch = shapes_data[choice(train_size, batch_size, replace=False), :]
                session.run(training_op, feed_dict={X: X_batch})
            loss_val, reconstruction_loss_val, latent_loss_val = \
                session.run([loss, reconstruction_loss, latent_loss], feed_dict={X: X_batch})

            logger.info('epoch: %s  %.4f %.4f %.4f' % (epoch, loss_val, reconstruction_loss_val, latent_loss_val))
            saver.save(session, model_checkpoint)
        codings_rnd = normal(size=[10, n_hidden3])
        outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})

    # show_reconstructed(X, outputs, model_checkpoint, n_test_samples=test_size, arg_shape=image_shape)
    # save_fig('reconstruction_plot', output_folder, arg_logger=logger, tight_layout=True)
    for iteration in range(10):
        plt.subplot(10, 10, iteration + 1)
        plot_image(outputs_val[iteration])

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
