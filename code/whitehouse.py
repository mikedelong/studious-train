# https://medium.com/district-data-labs/basics-of-entity-resolution-with-python-and-dedupe-bc87440b64d4
import logging
from os.path import exists
from time import time

import pandas as pd

if __name__ == '__main__':
    start_time = time()

    console_formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    file_formatter = logging.Formatter('%(asctime)s : %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.info('started')

    data_folder = '../data/'
    output_folder = '../output/'

    url = 'https://raw.githubusercontent.com/maxharlow/tutorials/master/' \
          'find-connections-with-fuzzy-matching/white-house-visitors.csv'
    csv_file = 'white-house-visitors.csv'
    # todo does this introduce any extra columns etc?
    input_file = data_folder + csv_file
    if not exists(input_file):
        logger.info('reading data from URL {}'.format(url))
        input_df = pd.read_csv(url)
        input_df.to_csv(input_file, index=False)

    logger.info('loading data from {}'.format(input_file))
    input_df = pd.read_csv(input_file)
    logger.info('our raw input data has {} rows and {} columns'.format(len(input_df), len(list(input_df))))
    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
