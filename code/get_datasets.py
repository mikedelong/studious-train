# http://scikit-learn.org/stable/datasets/index.html
import logging
from time import time
from urllib.error import HTTPError
from warnings import catch_warnings
from warnings import filterwarnings

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_kddcup99
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_mldata
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

    logger.info('loading book evaluation data')
    book_evaluation_bunch = fetch_mldata('book-evaluation-complete', data_home=data_folder)
    book_data = book_evaluation_bunch['data']
    logger.info('data is %d x %d' % book_data.shape)
    book_column_names = book_evaluation_bunch['COL_NAMES']
    logger.info('book evaluation column names: %s' % book_column_names)
    book_description = book_evaluation_bunch['DESCR']
    logger.info('book evaluation description: %s' % book_description)

    logger.info('loading boston data')
    boston_bunch = load_boston(return_X_y=return_X_y)
    boston_data = boston_bunch.data
    logger.info('boston data is %d x %d' % boston_data.shape)
    boston_target = boston_bunch.target
    boston_feature_names = boston_bunch.feature_names
    logger.info('boston feature names: %s' % boston_feature_names)
    boston_description = boston_bunch.DESCR
    logger.debug('boston description: %s' % boston_description)

    logger.info('loading breast cancer data')
    cancer_bunch = load_breast_cancer(return_X_y=return_X_y)
    cancer_data = cancer_bunch['data']
    logger.info('cancer data is %d x %d' % cancer_data.shape)
    cancer_target = cancer_bunch['target']
    cancer_feature_names = cancer_bunch['feature_names']
    logger.info('cancer feature names are %s' % cancer_feature_names)
    cancer_description = cancer_bunch['DESCR']
    logger.debug('cancer description: %s' % cancer_description)

    logger.info('loading diabetes data')
    diabetes_bunch = load_diabetes(return_X_y=return_X_y)
    diabetes_data = diabetes_bunch['data']
    logger.info('diabetes data is %d x %d' % diabetes_data.shape)
    diabetes_target = diabetes_bunch['target']
    diabetes_feature_names = diabetes_bunch['feature_names']
    logger.info('diabetes feature names are %s' % diabetes_feature_names)
    diabetes_description = diabetes_bunch['DESCR']
    logger.debug('diabetes description: %s' % diabetes_description)

    logger.info('loading digits data')
    digits_bunch = load_digits(return_X_y=return_X_y)
    digits_data = digits_bunch['data']
    logger.info('digits data is %d x %d' % diabetes_data.shape)
    digits_target = digits_bunch['target']
    digits_target_names = digits_bunch['target_names']
    logger.debug('digits target names are %s' % digits_target_names)
    digits_images = digits_bunch['images']
    digits_description = digits_bunch['DESCR']
    logger.debug('digits description: %s' % digits_description)

    logger.info('loading fish killer data')
    try:
        fish_killer_bunch = fetch_mldata('fish_killer', data_home=data_folder)
        fish_killer_column_names = fish_killer_bunch['COL_NAMES']
        logger.info('fish killer column names: %s' % fish_killer_column_names)
        fish_killer_data = fish_killer_bunch['data']
        logger.info('fish killer data is %d x %d' % fish_killer_data.shape)
        fish_killer_description = fish_killer_bunch['DESCR']
        fish_killer_int2 = fish_killer_bunch['int2']
        fish_killer_target = fish_killer_bunch['target']
        logger.info('fish killer target variable: %s' % fish_killer_target)
    except HTTPError as httpError:
        logger.warning(httpError)

    logger.info('loading industry portfolio data')
    try:
        industry_portfolio_bunch = fetch_mldata('industry-portfolio', data_home=data_folder)
        industry_portfolio_column_names = industry_portfolio_bunch['COL_NAMES']
        logger.info('industry portfolio column names are %s' % industry_portfolio_column_names)
        industry_portfolio_data = industry_portfolio_bunch['data']
        logger.info('industry portfolio data is %d x %d' % industry_portfolio_data.shape)
        industry_portfolio_description = industry_portfolio_bunch['DESCR']
        logger.debug('industry portfolio description: %s' % industry_portfolio_description)
    except HTTPError as httpError:
        logger.warning(httpError)

    logger.info('loading iris data')
    iris_bunch = load_iris(return_X_y=return_X_y)
    iris_data = iris_bunch['data']
    logger.info('iris data is %d x %d' % iris_data.shape)
    iris_target = iris_bunch['target']
    iris_target_names = iris_bunch['target_names']
    logger.info('iris target names: %s' % iris_target_names)
    iris_feature_names = iris_bunch['feature_names']
    logger.info('iris feature names: %s' % iris_feature_names)
    iris_description = iris_bunch['DESCR']
    logger.debug('iris description: %s' % iris_description)

    random_state = 1
    logger.info('loading KDD data')
    kdd_bunch = fetch_kddcup99(random_state=random_state, download_if_missing=True)
    kdd_data = kdd_bunch['data']
    logger.info('KDD data is %d x %d' % kdd_data.shape)
    kdd_target = kdd_bunch['target']
    logger.info('KDD target unique values: %s' % list(set(kdd_target)))

    min_faces_per_person = 0
    lfw_resize = None
    logger.info('loading LFW people')
    with catch_warnings():
        filterwarnings('ignore', category=DeprecationWarning)
        lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=lfw_resize)
    lfw_people_data = lfw_people['data']
    logger.info('LFW data is %d x %d' % lfw_people_data.shape)
    lfw_people_images = lfw_people['images']
    lfw_people_target = lfw_people['target']
    logger.info('LFW target is %s' % lfw_people_target)
    lfw_people_target_names = lfw_people['target_names']
    lfw_people_description = lfw_people['DESCR']
    logger.info('the LFW data is %d x %d' % lfw_people_data.shape)

    logger.info('loading Linnerud data')
    linnerud_bunch = load_linnerud(return_X_y=return_X_y)
    linnerud_data = linnerud_bunch['data']
    linnerud_feature_names = linnerud_bunch['feature_names']
    linnerud_target = linnerud_bunch['target']
    linnerud_target_names = linnerud_bunch['target_names']
    linnerud_description = linnerud_bunch['DESCR']

    logger.info('loading newsgroups data')
    newsgroups_bunch = fetch_20newsgroups(data_home=data_folder)
    newsgroups_data = newsgroups_bunch['data']
    newsgroups_target_names = newsgroups_bunch['target_names']
    newsgroups_description = newsgroups_bunch['description']
    newsgroups_target = newsgroups_bunch['target']
    newsgroups_filenames = newsgroups_bunch['filenames']

    try:
        nile_water_level_bunch = fetch_mldata('nile-water-level', data_home=data_folder)
        nile_water_level_data = nile_water_level_bunch['data']
        logger.info('nile water level data is %d x %d' % nile_water_level_data.shape)
        nile_water_level_description = nile_water_level_bunch['DESCR']
        nile_water_level_column_names = nile_water_level_bunch['COL_NAMES']
        logger.info('nile water level column names: %s' % nile_water_level_column_names)
    except HTTPError as httpError:
        logger.warning(httpError)

    logger.info('loading Olivetti faces data')
    olivetti_faces = fetch_olivetti_faces(data_home=data_folder)
    olivetti_faces_data = olivetti_faces['data']
    olivetti_faces_images = olivetti_faces['images']
    olivetti_faces_target = olivetti_faces['target']
    olivetti_faces_description = olivetti_faces['DESCR']

    logger.info('loading Reuters Corpus Volume I data')
    rcv1_bunch = fetch_rcv1(subset='all', download_if_missing=True, random_state=random_state)
    rcv1_data = rcv1_bunch['data']
    rcv1_target = rcv1_bunch['target']
    rcv1_sample_id = rcv1_bunch['sample_id']
    rcv1_target_names = rcv1_bunch['target_names']
    rcv1_description = rcv1_bunch['DESCR']

    logger.info('loading sample images data')
    with catch_warnings():
        filterwarnings('ignore', category=DeprecationWarning)
        sample_images_bunch = load_sample_images()
    sample_images = sample_images_bunch['images']
    sample_images_filenames = sample_images_bunch['filenames']
    sample_images_description = sample_images_bunch['DESCR']

    logger.info('loading well log data')
    try:
        well_log_bunch = fetch_mldata('well-log', data_home=data_folder)
        well_log_column_names = well_log_bunch['COL_NAMES']
        logger.info('well log column names: %s' % well_log_column_names)
        well_log_data = well_log_bunch['data']
        logger.info('well log data is %d x %d' % well_log_data.shape)
    except HTTPError as httpError:
        logger.warning(httpError)

    # this one is garbage because the data is full of NaNs
    logger.info('loading Whistler daily snowfall data')
    try:
        whistler_daily_bunch = fetch_mldata('whistler-daily-snowfall', data_home=data_folder)
        whistler_data = whistler_daily_bunch['data']
        logger.info('Whistler daily snowfall data is %d x %d' % whistler_data.shape)
    except HTTPError as httpError:
        logger.warning(httpError)

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
