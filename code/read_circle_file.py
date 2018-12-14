import logging
from glob import glob
from pickle import dump
from time import time

from PIL import Image
from PIL import ImageOps
from numpy import array

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

    input_file = '../output/circle_frames/plotcircle21.png'
    image = array(ImageOps.invert(Image.open(input_file).convert('L')))
    logger.info(image.shape)

    result = list()
    for input_file in glob('../output/circle_frames/*.png'):
        logger.info('processing file %s' % input_file)
        image = array(ImageOps.invert(Image.open(input_file).convert('L')))
        result.append(image)
    data_circles_pkl = '../data/circles.pkl'
    logger.info('writing circles data to %s' % data_circles_pkl)
    with open(data_circles_pkl, 'wb') as circles_fp:
        dump(result, circles_fp)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
