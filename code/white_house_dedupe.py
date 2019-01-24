# https://medium.com/district-data-labs/basics-of-entity-resolution-with-python-and-dedupe-bc87440b64d4
import logging
from os.path import exists
from time import time

import dedupe
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
    input_file = data_folder + csv_file
    if not exists(input_file):
        logger.info('reading data from URL {}'.format(url))
        input_df = pd.read_csv(url)
        input_df.to_csv(input_file, index=False)

    nrows = 1000
    logger.info('loading data from {}'.format(input_file))
    input_df = pd.read_csv(input_file, nrows=nrows)
    # clean up the column names
    input_df.columns = [item.strip() for item in input_df.columns]
    logger.info(list(input_df))
    # input_df['date'] = input_df['date'].astype('datetime64[ms]')
    for column in list(input_df):  # [item for item in list(input_df) if item != 'date']:
        logger.info(column)
        input_df[column] = input_df[column].astype(str)
        input_df[column] = input_df[column].apply(lambda x: x.strip())

    # it turns out the gender column is no good
    input_df = input_df.drop(['gender'], axis=1)
    logger.info(input_df['race'].unique())

    logger.info('our raw input data has {} rows and {} columns'.format(len(input_df), len(list(input_df))))
    logger.info('our data has columns {}'.format(list(input_df)))
    logger.info('if we remove duplicates we have only {} rows'.format(len(input_df.drop_duplicates())))

    input_dict = {index: value for index, value in enumerate(input_df.to_dict(orient='records'))}

    training_file = './whitehouse_dedupe_training.json'
    fields = [
        {'field': 'visitor_name', 'type': 'String'},
        {'field': 'race', 'type': 'String', 'has missing': True},
    ]
    deduper = dedupe.Dedupe(fields)
    deduper.sample(input_dict, 30)  # let's start with 20 because apparently 100 is too many
    dedupe.consoleLabel(deduper)
    deduper.train()
    logger.info('writing training results to {}'.format(training_file))
    with open(training_file, 'w') as training_fp:
        deduper.writeTraining(training_fp)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
