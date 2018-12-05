# https://raw.githubusercontent.com/datasets/world-cities/master/data/world-cities.csv
import logging
from time import time

import pandas as pd

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

    usecols = ['name']
    data_df = pd.read_csv('../data/world-cities.csv', usecols=usecols)
    logger.info(list(data_df))
    logger.info(data_df.shape)
    logger.info(data_df.tail())
    output_file = '../output/world_cities.csv'
    logger.info('writing result to %s' % output_file)
    data_df.to_csv(output_file, index=False)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
