import logging
from glob import glob
from pickle import dump
from random import seed
from random import shuffle
from time import time

from PIL import Image
from PIL import ImageOps
from numpy import array
from numpy import stack

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

    # load everything up in a list comprehension and do the processing in-line
    result = [array(ImageOps.invert(Image.open(input_file).convert('L'))).flatten() for input_file in
              glob('../output/circle_frames/*.png')]

    # shuffle in place
    random_seed = 1
    seed(random_seed)
    shuffle(result)

    data_circles_pkl = '../data/circles.pkl'
    logger.info('writing circles data to %s' % data_circles_pkl)
    with open(data_circles_pkl, 'wb') as circles_fp:
        dump(stack(result, axis=0), circles_fp)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
