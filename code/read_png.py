import logging
from glob import glob
from time import time

import numpy as np
from imageio import imread


def mse(left, right):
    # is this really the best way to do this?
    err = np.sum((left.astype('float') - right.astype('float')) ** 2)
    err /= float(left.shape[0] * left.shape[1])
    return err


def get_ply(arg_list, arg_index):
    return [image[:, :, arg_index] for image in arg_list]


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
    input_folder = '../../../data/MoonPix/'

    if False:
        for item in glob(input_folder + '*.jpg'):
            image = imread(item)
            logger.info('%s %s' % (item, image.shape))

    limit = 20
    images = [
        imread(item) for index, item in enumerate(glob(input_folder + '*.jpg')) if index < limit
    ]
    logger.info('we read %d images and each one is %d x %d x %d' % (
        len(images), images[0].shape[0], images[0].shape[1], images[0].shape[2]))

    if images[0].shape[2] != 3:
        logger.warning('we only support 3 channels; quitting')
        quit()
    images_blue = get_ply(images, 0)
    images_green = get_ply(images, 1)
    images_red = get_ply(images, 2)

    # todo loop over the whole space
    for index, image in enumerate(images):
        if index == 0:
            pass
        else:
            logger.info(mse(images[index - 1], image))

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
