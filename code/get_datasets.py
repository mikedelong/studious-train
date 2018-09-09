import logging
from time import time
from warnings import catch_warnings
from warnings import filterwarnings

from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_olivetti_faces
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

    logger.debug('loading boston data')
    boston_bunch = load_boston(return_X_y=return_X_y)
    boston_data = boston_bunch.data
    logger.info('boston data is %d x %d' % boston_data.shape)
    boston_target = boston_bunch.target
    boston_feature_names = boston_bunch.feature_names
    boston_description = boston_bunch.DESCR

    cancer_bunch = load_breast_cancer(return_X_y=return_X_y)
    cancer_data = cancer_bunch['data']
    cancer_target = cancer_bunch['target']
    cancer_feature_names = cancer_bunch['feature_names']
    cancer_description = cancer_bunch['DESCR']

    diabetes_bunch = load_diabetes(return_X_y=return_X_y)
    diabetes_data = diabetes_bunch['data']
    diabetes_target = diabetes_bunch['target']
    diabetes_feature_names = diabetes_bunch['feature_names']
    diabetes_description = diabetes_bunch['DESCR']

    digits_bunch = load_digits(return_X_y=return_X_y)
    digits_data = digits_bunch['data']
    digits_target = digits_bunch['target']
    digits_target_names = digits_bunch['target_names']
    digits_images = digits_bunch['images']
    digits_description = digits_bunch['DESCR']

    iris_bunch = load_iris(return_X_y=return_X_y)
    iris_data = iris_bunch['data']
    iris_target = iris_bunch['target']
    iris_target_names = iris_bunch['target_names']
    iris_feature_names = iris_bunch['feature_names']
    iris_description = iris_bunch['DESCR']

    min_faces_per_person = 0
    lfw_resize = None
    logger.info('downloading LFW people')
    with catch_warnings():
        filterwarnings("ignore", category=DeprecationWarning)
        lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=lfw_resize)
    lfw_people_data = lfw_people['data']
    lfw_people_images = lfw_people['images']
    lfw_people_target = lfw_people['target']
    lfw_people_target_names = lfw_people['target_names']
    lfw_people_description = lfw_people['DESCR']
    logger.info(lfw_people_data.shape)

    linnerud_bunch = load_linnerud(return_X_y=return_X_y)
    linnerud_data = linnerud_bunch['data']
    linnerud_feature_names = linnerud_bunch['feature_names']
    linnerud_target = linnerud_bunch['target']
    linnerud_target_names = linnerud_bunch['target_names']
    linnerud_description = linnerud_bunch['DESCR']

    newsgroups_bunch = fetch_20newsgroups(data_home=data_folder)
    newsgroups_data = newsgroups_bunch['data']
    newsgroups_target_names = newsgroups_bunch['target_names']
    newsgroups_description = newsgroups_bunch['description']
    newsgroups_target = newsgroups_bunch['target']
    newsgroups_filenames = newsgroups_bunch['filenames']

    olivetti_faces = fetch_olivetti_faces(data_home=data_folder)
    olivetti_faces_data = olivetti_faces['data']
    olivetti_faces_images = olivetti_faces['images']
    olivetti_faces_target = olivetti_faces['target']
    olivetti_faces_description = olivetti_faces['DESCR']

    sample_images_bunch = load_sample_images()
    sample_images = sample_images_bunch['images']
    sample_images_filenames = sample_images_bunch['filenames']
    sample_images_description = sample_images_bunch['DESCR']

    wine_bunch = load_wine(return_X_y=return_X_y)
    wine_data = wine_bunch['data']
    wine_target = wine_bunch['target']
    wine_target_names = wine_bunch['target_names']
    wine_feature_names = wine_bunch['feature_names']
    wine_description = wine_bunch['DESCR']

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
