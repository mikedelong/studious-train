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

    # todo break this into 3 components/colors
    # todo loop over the whole space
    logger.info('we read %d images', len(images))
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
