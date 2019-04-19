# http://scikit-learn.org/stable/datasets/index.html
# https://www.statsmodels.org/dev/datasets/index.html
import logging
import pickle
from os import mkdir
from os.path import exists
from time import time
from warnings import catch_warnings
from warnings import filterwarnings

from pandas import read_csv
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_kddcup99
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_rcv1
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.datasets import load_linnerud
from sklearn.datasets import load_sample_images
from sklearn.datasets import load_wine
from statsmodels.datasets import anes96
from statsmodels.datasets import cancer
from statsmodels.datasets import ccard
from statsmodels.datasets import china_smoking
from statsmodels.datasets import co2
from statsmodels.datasets import committee
from statsmodels.datasets import copper
from statsmodels.datasets import cpunish
from statsmodels.datasets import elnino
from statsmodels.datasets import engel
from statsmodels.datasets import fair
from statsmodels.datasets import fertility
from statsmodels.datasets import get_rdataset
from statsmodels.datasets import grunfeld
from statsmodels.datasets import heart
from statsmodels.datasets import interest_inflation
from statsmodels.datasets import longley
from statsmodels.datasets import macrodata
from statsmodels.datasets import modechoice
from statsmodels.datasets import nile
from statsmodels.datasets import randhie
from statsmodels.datasets import scotland
from statsmodels.datasets import spector
from statsmodels.datasets import stackloss
from statsmodels.datasets import star98
from statsmodels.datasets import statecrime
from statsmodels.datasets import strikes
from statsmodels.datasets import sunspots

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

    documentation_file = '../documentation/datasets.csv'
    datasets_usecols = ['Package', 'Item']
    datasets_df = read_csv(documentation_file, usecols=datasets_usecols)
    logger.info(datasets_df.shape)
    packages = datasets_df['Package'].unique()
    for package in packages:
        if not exists(data_folder + package):
            logger.info('creating output folder {}'.format(data_folder + package))
            mkdir(data_folder + package)

    for index, row in datasets_df.iterrows():
        logger.info('loading {} / {} data'.format(row['Package'], row['Item']))
        current_pickle = data_folder + row['Package'] + '/' + row['Item'] + '.pkl'
        if exists(current_pickle):
            with open(current_pickle, 'rb') as current_fp:
                current_bundle = pickle.load(current_fp)
        else:
            current_bundle = get_rdataset(row['Item'], row['Package'])
            with open(current_pickle, 'wb') as current_fp:
                pickle.dump(current_bundle, current_fp)
        current_data = current_bundle.data
        logger.info('{} data has variables {}'.format(row['Item'], list(current_data)))
        if len(current_data.shape) > 1:
            logger.info('{} data has {} rows and {} variables'.format(row['Item'], current_data.shape[0],
                                                                      current_data.shape[1]))
        if 'title' in current_bundle.keys():
            current_title = current_bundle.title
            logger.info('{} data has title {}'.format(row['Item'], current_title))

    return_X_y = False

    if not exists(data_folder + 'statsmodels'):
        logger.info('creating output folder {}'.format(data_folder + 'statsmodels'))
        mkdir(data_folder + 'statsmodels')

    for key, value in {
        'anes96': anes96,
        'cancer': cancer,
        'ccard': ccard,
        'china_smoking': china_smoking,
        # 'co2': co2,
        'committee': committee,
        'copper': copper,
        'cpunish': cpunish,
        'elnino': elnino,
        'engel': engel,
        'fair': fair,
        'fertility': fertility,
        'grunfeld': grunfeld,
        'heart': heart,
        'interest_inflation': interest_inflation,
        'longley': longley,
        'macrodata': macrodata,
        'modechoice': modechoice,
        'nile': nile,
        'randhie': randhie,
        'scotland': scotland,
        'spector': spector,
        'stackloss': stackloss,
        'star98': star98,
        'statecrime': statecrime,
        'strikes': strikes,
        'sunspots': sunspots
    }.items():
        logger.info('loading {} data'.format(key))
        current_pickle = data_folder + 'statsmodels' + '/' + '{}.pkl'.format(key)
        if exists(current_pickle):
            with open(current_pickle, 'rb') as current_fp:
                current_bunch = pickle.load(current_fp)
        else:
            current_bunch = value.load_pandas()
            with open(current_pickle, 'wb') as current_fp:
                pickle.dump(current_bunch, current_fp)
        current_data = current_bunch['data']
        logger.info('{} data is {} x {}'.format(key, current_data.shape[0], current_data.shape[1]))
        if 'names' in current_bunch.keys():
            current_names = current_bunch['names']
        if 'endog_name' in current_bunch.keys():
            logger.info('{} endogenous variable is {}'.format(key, current_bunch['endog_name']))
        if 'exog_name' in current_bunch.keys():
            logger.info('{} exogenous variable is %s' % current_bunch['exog_name'])

    if not exists(data_folder + 'sklearn'):
        logger.info('creating output folder {}'.format(data_folder + 'sklearn'))
        mkdir(data_folder + 'sklearn')

    for key, value in {
        'boston': load_boston,
        'breast_cancer': load_breast_cancer,
        'diabetes': load_diabetes,
        'digits': load_digits,
        'iris': load_iris,
        'linnerud': load_linnerud,
        'wine': load_wine
    }.items():
        logger.info('loading {} data'.format(key))
        current_pickle = data_folder + '{}/{}.pkl'.format('sklearn', key)
        if exists(current_pickle):
            with open(current_pickle, 'rb') as current_fp:
                current_bunch = pickle.load(current_fp)
        else:
            current_bunch = value(return_X_y=return_X_y)
            with open(current_pickle, 'wb') as current_fp:
                pickle.dump(current_bunch, current_fp)
        current_data = current_bunch.data
        logger.info('boston data is %d x %d' % current_data.shape)
        current_target = current_bunch.target
        if 'feature_names' in current_bunch.keys():
            current_feature_names = current_bunch.feature_names
            logger.info('{} feature names: {}'.format(key, current_feature_names))
        current_description = current_bunch.DESCR
        logger.debug('{} description: {}'.format(key, current_description))

    logger.info('loading CO2 data')
    co2_pickle = data_folder + 'co2.pkl'
    if exists(co2_pickle):
        with open(co2_pickle, 'rb') as co2_fp:
            co2_bunch = pickle.load(co2_fp)
    else:
        co2_bunch = co2.load()
        with open(co2_pickle, 'wb') as co2_fp:
            pickle.dump(co2_bunch, co2_fp)
    co2_data = co2_bunch['data']
    logger.info('CO2 data has %d rows' % len(co2_data))
    co2_names = co2_bunch['names']
    logger.info('CO2 names: %s' % str(co2_names))
    co2_raw_data = co2_bunch['raw_data']
    logger.info('CO2 raw data is %d x %d' % co2_raw_data.shape)

    logger.info('loading KDD data')
    random_state = 1
    kdd_pickle = data_folder + 'kddcup99.pkl'
    if exists(kdd_pickle):
        with open(kdd_pickle, 'rb') as kdd_fp:
            kdd_bunch = pickle.load(kdd_fp)
    else:
        kdd_bunch = fetch_kddcup99(random_state=random_state, download_if_missing=True)
        with open(kdd_pickle, 'wb') as kdd_fp:
            pickle.dump(kdd_bunch, kdd_fp)
    kdd_data = kdd_bunch['data']
    logger.info('KDD data is %d x %d' % kdd_data.shape)
    kdd_target = kdd_bunch['target']
    logger.info('KDD target unique values: %s' % list(set(kdd_target)))

    min_faces_per_person = 0
    lfw_resize = None
    lfw_pickle = data_folder + 'lfw.pkl'
    logger.info('loading LFW people')
    if exists(lfw_pickle):
        with open(lfw_pickle, 'rb') as lfw_fp:
            lfw_people = pickle.load(lfw_fp)
    else:
        with catch_warnings():
            filterwarnings('ignore', category=DeprecationWarning)
            lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=lfw_resize)
            with open(lfw_pickle, 'wb') as lfw_fp:
                pickle.dump(lfw_people, lfw_fp)
    lfw_people_data = lfw_people['data']
    logger.info('LFW data is %d x %d' % lfw_people_data.shape)
    lfw_people_images = lfw_people['images']
    lfw_people_target = lfw_people['target']
    logger.info('LFW target is %s' % lfw_people_target)
    lfw_people_target_names = lfw_people['target_names']
    lfw_people_description = lfw_people['DESCR']
    logger.info('the LFW data is %d x %d' % lfw_people_data.shape)

    # todo add pickle file
    logger.info('loading newsgroups data')
    newsgroups_bunch = fetch_20newsgroups(data_home=data_folder)
    newsgroups_data = newsgroups_bunch['data']
    newsgroups_target_names = newsgroups_bunch['target_names']
    newsgroups_description = newsgroups_bunch['DESCR']
    newsgroups_target = newsgroups_bunch['target']
    newsgroups_filenames = newsgroups_bunch['filenames']

    logger.info('loading Olivetti faces data')
    olivetti_faces = fetch_olivetti_faces(data_home=data_folder)
    olivetti_faces_data = olivetti_faces['data']
    olivetti_faces_images = olivetti_faces['images']
    olivetti_faces_target = olivetti_faces['target']
    olivetti_faces_description = olivetti_faces['DESCR']

    logger.info('loading Reuters Corpus Volume I data')
    reuters_pickle = data_folder + 'reuters.pkl'
    rcv1_bunch = fetch_rcv1(subset='all', download_if_missing=True, random_state=random_state)
    rcv1_data = rcv1_bunch['data']
    rcv1_target = rcv1_bunch['target']
    rcv1_sample_id = rcv1_bunch['sample_id']
    rcv1_target_names = rcv1_bunch['target_names']
    rcv1_description = rcv1_bunch['DESCR']
    logger.info('Reuters data has description %s' % str(rcv1_description).strip())

    logger.info('loading sample images data')
    with catch_warnings():
        filterwarnings('ignore', category=DeprecationWarning)
        sample_images_bunch = load_sample_images()
    sample_images = sample_images_bunch['images']
    sample_images_filenames = sample_images_bunch['filenames']
    sample_images_description = sample_images_bunch['DESCR']

    # todo: refactor
    logger.info('loading wine data')
    wine_bunch = load_wine(return_X_y=return_X_y)
    wine_data = wine_bunch['data']
    logger.info('wine data is %d x %d' % wine_data.shape)
    wine_target = wine_bunch['target']
    wine_target_names = wine_bunch['target_names']
    logger.info('wine target names: %s' % wine_target_names)
    wine_feature_names = wine_bunch['feature_names']
    logger.info('wine feature names: %s' % wine_feature_names)
    wine_description = wine_bunch['DESCR']

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
