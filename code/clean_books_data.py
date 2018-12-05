import logging
from time import time

import pandas as pd


def modify_author(arg):
    if ',' not in arg:
        return arg
    else:
        (piece0, piece1) = arg.split(',')
        return ' '.join([piece1, piece0]).strip()


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

    usecols = ['Title', 'Author']
    source_df = pd.read_csv('../data/books.csv', usecols=usecols)
    source_df = source_df.dropna()

    source_df['modified_author'] = source_df['Author'].apply(modify_author)
    source_df = source_df.drop(['Author'], axis=1)
    source_df = source_df.rename(mapper={'modified_author': 'Author'}, axis=1)
    source_df.to_csv('../data/books_clean.csv', sep=';', index=False)
    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
