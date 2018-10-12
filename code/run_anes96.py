# http://scikit-learn.org/stable/datasets/index.html
# https://www.statsmodels.org/dev/datasets/index.html
import logging
import pickle
from os.path import exists
from time import time

from statsmodels.datasets import anes96

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
    return_X_y = False

    logger.info('loading ANES96 data')
    anes96_pickle = data_folder + 'anes96.pkl'
    if exists(anes96_pickle):
        with open(anes96_pickle, 'rb') as anes96_fp:
            anes96_bunch = pickle.load(anes96_fp)
    else:
        anes96_bunch = anes96.load_pandas()
        with open(anes96_pickle, 'wb') as anes96_fp:
            pickle.dump(anes96_bunch, anes96_fp)
    anes96_data = anes96_bunch['data']
    logger.info('ANES96 data is %d x %d' % anes96_data.shape)
    anes96_names = anes96_bunch['names']
    anes96_endog = anes96_bunch['endog_name']
    logger.info('ANES96 endogengous variable is %s' % anes96_endog)
    anes96_exog = anes96_bunch['exog_name']
    logger.info('ANES96 exogengous variable is %s' % anes96_exog)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
