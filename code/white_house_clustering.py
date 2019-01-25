import logging
from os.path import exists
from time import time

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import jaccard_similarity_score

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

    nrows = 400
    logger.info('loading data from {}'.format(input_file))
    input_df = pd.read_csv(input_file, nrows=nrows)
    # clean up the column names
    input_df.columns = [item.strip() for item in input_df.columns]
    logger.info(list(input_df))
    for column in list(input_df):  # [item for item in list(input_df) if item != 'date']:
        logger.info('converting column {} to string'.format(column))
        input_df[column] = input_df[column].astype(str)
        input_df[column] = input_df[column].apply(lambda x: x.strip())

    # it turns out the gender column is no good
    logger.info('dropping gender column because it contains the duplicate of race data')
    input_df = input_df.drop(['gender'], axis=1)
    for column in list(input_df):
        logger.info('column {} has {} unique values'.format(column, input_df[column].nunique()))

    logger.info('our raw input data has {} rows and {} columns'.format(len(input_df), len(list(input_df))))
    logger.info('our data has columns {}'.format(list(input_df)))
    logger.info('if we remove duplicates we have only {} rows'.format(len(input_df.drop_duplicates())))

    names = input_df.drop_duplicates()['visitor_name'].unique()
    logger.info(names)
    # http://sujitpal.blogspot.com/2018/08/keyword-deduplication-using-python.html

    hasher = FeatureHasher(input_type="string", n_features=25, dtype=np.int32)

    hashes = [
        hasher.transform([[''.join(trigram) for trigram in nltk.trigrams([c for c in name])]]).toarray()
        for name in names
    ]

    scores = [[jaccard_similarity_score(hashes[i][0], hashes[j][0])
               for i in range(len(names))] for j in range(len(names))]

    scores = np.array(scores)
    scores_df = pd.DataFrame(data=scores, index=names, columns=names)
    pdist_result = pdist(scores)
    linkage_result = linkage(pdist_result, method='complete')
    index = fcluster(linkage_result, 0.5 * pdist_result.max(), 'distance')
    dendrogram(linkage_result)
    plt.show()

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
