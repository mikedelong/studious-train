import logging
import os
from functools import partial
from pickle import load
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.random import choice


def save_fig(fig_id, arg_folder, arg_logger, tight_layout=True):
    suffix = '.png'
    path = os.path.join(arg_folder, fig_id + suffix)
    arg_logger.info('Saving figure %s' % fig_id + suffix)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def plot_multiple_images(images, arg_nrows, arg_ncols, pad=2):
    images = images - images.min()  # make the minimum == 0, so the padding looks white
    w, h = images.shape[1:]
    image = np.zeros(((w + pad) * arg_nrows + pad, (h + pad) * arg_ncols + pad))
    for y in range(arg_nrows):
        for x in range(arg_ncols):
            image[(y * (h + pad) + pad):(y * (h + pad) + pad + h), (x * (w + pad) + pad):(x * (w + pad) + pad + w)] = \
                images[y * arg_ncols + x]
    plt.imshow(image, cmap='Greys', interpolation='nearest')
    plt.axis('off')


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

    shapes_file = '../data/rectangles.pkl'
    with open(shapes_file, 'rb') as shapes_fp:
        shapes_data = load(shapes_fp)
    logger.info('loaded %d items from %s' % (len(shapes_data), shapes_file))

    # we need to convert the data to floats and normalize to get the machinery below to work properly
    shapes_data = shapes_data.astype('float')
    shapes_data = np.multiply(shapes_data, 1.0 / shapes_data.max())
    logger.info('the training data has max %.4f and min %.4f' % (shapes_data.max(), shapes_data.min()))

    n_inputs = 32 * 32
    n_hidden1 = 500
    n_hidden2 = 500
    n_hidden3 = 20  # was 20
    n_hidden4 = n_hidden2
    n_hidden5 = n_hidden1
    n_outputs = n_inputs
    learning_rate = 0.001

    initializer = tf.contrib.layers.variance_scaling_initializer()
    my_dense_layer = partial(tf.layers.dense, activation=tf.nn.elu, kernel_initializer=initializer)

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
    latent_loss = 0.5 * tf.reduce_sum(tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)
    loss = reconstruction_loss + latent_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 50
    n_samples = 16
    batch_size = 3000
    n_rows = 4
    n_cols = 4
    model_checkpoint = '../models/variational_rectangle_generator.ckpt'

    num_examples = 10000
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            n_batches = num_examples // batch_size
            X_batch = None
            for iteration in range(n_batches):
                X_batch = shapes_data[choice(num_examples, batch_size, replace=False), :]
                sess.run(training_op, feed_dict={X: X_batch})
            loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss],
                                                                          feed_dict={X: X_batch})
            logger.info('epoch: %s loss %.4f recon %.4f latent %.4f' % (
                epoch, loss_val, reconstruction_loss_val, latent_loss_val))
            saver.save(sess, model_checkpoint)

            codings_rnd = np.random.normal(size=[n_samples, n_hidden3])
            outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})

            plot_multiple_images(outputs_val.reshape(-1, 32, 32), n_rows, n_cols)
            d__format = 'generated_plot{:03d}'.format(epoch)
            save_fig(d__format, arg_logger=logger, arg_folder='../output/variational_rectangles/')

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
