# http://scikit-learn.org/stable/datasets/index.html
# https://www.statsmodels.org/dev/datasets/index.html
import logging
import pickle
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
    for index, row in datasets_df.iterrows():
        logger.info('loading {} / {} data'.format(row['Package'], row['Item']))
        current_pickle = data_folder + row['Item'] + '.pkl'
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

    # todo add pickle file
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
    breast_cancer_bunch = load_breast_cancer(return_X_y=return_X_y)
    breast_cancer_data = breast_cancer_bunch['data']
    logger.info('cancer data is %d x %d' % breast_cancer_data.shape)
    breast_cancer_target = breast_cancer_bunch['target']
    breast_cancer_feature_names = breast_cancer_bunch['feature_names']
    logger.info('cancer feature names are %s' % breast_cancer_feature_names)
    breast_cancer_description = breast_cancer_bunch['DESCR']
    logger.debug('cancer description: %s' % breast_cancer_description)

    logger.info('loading cancer data')
    cancer_pickle = data_folder + 'cancer.pkl'
    if exists(cancer_pickle):
        with open(cancer_pickle, 'rb') as cancer_fp:
            cancer_bunch = cancer.load()
    else:
        cancer_bunch = cancer.load_pandas()
        with open(cancer_pickle, 'wb') as cancer_fp:
            pickle.dump(cancer_bunch, cancer_fp)
    cancer_data = cancer_bunch['data']
    logger.info('Cancer data is 2 x %d' % len(cancer_data))
    cancer_names = cancer_bunch['names']
    cancer_endog = cancer_bunch['endog_name']
    logger.info('Cancer endogenous variable is %s' % cancer_endog)
    cancer_exog = cancer_bunch['exog_name']
    logger.info('Cancer exogenous variable is %s' % cancer_exog)

    logger.info('loading ccard data')
    ccard_pickle = data_folder + 'ccard.pkl'
    if exists(ccard_pickle):
        with open(ccard_pickle, 'rb') as ccard_fp:
            ccard_bunch = pickle.load(ccard_fp)
    else:
        ccard_bunch = ccard.load()
        with open(ccard_pickle, 'wb') as ccard_fp:
            pickle.dump(ccard_bunch, ccard_fp)
    ccard_data = ccard_bunch['data']
    logger.info('ccard data has %d rows' % len(ccard_data))
    ccard_names = ccard_bunch['names']
    logger.info('ccard names: %s' % str(ccard_names))

    logger.info('loading china_smoking data')
    china_smoking_pickle = data_folder + 'china_smoking.pkl'
    if exists(china_smoking_pickle):
        with open(china_smoking_pickle, 'rb') as china_smoking_fp:
            china_smoking_bunch = pickle.load(china_smoking_fp)
    else:
        china_smoking_bunch = china_smoking.load()
        with open(china_smoking_pickle, 'wb') as china_smoking_fp:
            pickle.dump(china_smoking_bunch, china_smoking_fp)
    china_smoking_data = china_smoking_bunch['data']
    logger.info('china_smoking data has %d rows' % len(china_smoking_data))

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

    logger.info('loading committee data')
    committee_pickle = data_folder + 'committee.pkl'
    if exists(committee_pickle):
        with open(committee_pickle, 'rb') as committee_fp:
            committee_bunch = pickle.load(committee_fp)
    else:
        committee_bunch = committee.load()
        with open(committee_pickle, 'wb') as committee_fp:
            pickle.dump(committee_bunch, committee_fp)
    committee_data = committee_bunch['data']
    logger.info('committee data has %d rows' % len(committee_data))
    committee_names = committee_bunch['names']
    logger.info('committee names: %s' % str(committee_names))
    committee_raw_data = committee_bunch['raw_data']
    logger.info('committee raw data is %d x %d' % committee_raw_data.shape)

    logger.info('loading copper data')
    copper_pickle = data_folder + 'copper.pkl'
    if exists(copper_pickle):
        with open(copper_pickle, 'rb') as copper_fp:
            copper_bunch = pickle.load(copper_fp)
    else:
        copper_bunch = copper.load()
        with open(copper_pickle, 'wb') as copper_fp:
            pickle.dump(copper_bunch, copper_fp)
    copper_data = copper_bunch['data']
    logger.info('copper data has %d rows' % len(copper_data))
    copper_names = copper_bunch['names']
    logger.info('copper names: %s' % str(copper_names))
    copper_raw_data = copper_bunch['raw_data']
    logger.info('copper raw data is %d x %d' % copper_raw_data.shape)

    logger.info('loading capital punishment data')
    cpunish_pickle = data_folder + 'cpunish.pkl'
    if exists(cpunish_pickle):
        with open(cpunish_pickle, 'rb') as cpunish_fp:
            cpunish_bunch = pickle.load(cpunish_fp)
    else:
        cpunish_bunch = cpunish.load()
        with open(cpunish_pickle, 'wb') as cpunish_fp:
            pickle.dump(cpunish_bunch, cpunish_fp)
    cpunish_data = cpunish_bunch['data']
    logger.info('cpunish data has %d rows' % len(cpunish_data))
    cpunish_names = cpunish_bunch['names']
    logger.info('cpunish names: %s' % str(cpunish_names))
    cpunish_raw_data = cpunish_bunch['raw_data']
    logger.info('cpunish raw data is %d x %d' % cpunish_raw_data.shape)

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
    logger.info('digits data is %d x %d' % digits_data.shape)
    digits_target = digits_bunch['target']
    digits_target_names = digits_bunch['target_names']
    logger.debug('digits target names are %s' % digits_target_names)
    digits_images = digits_bunch['images']
    digits_description = digits_bunch['DESCR']
    logger.debug('digits description: %s' % digits_description)

    logger.info('loading el nino data')
    elnino_pickle = data_folder + 'elnino.pkl'
    if exists(elnino_pickle):
        with open(elnino_pickle, 'rb') as elnino_fp:
            elnino_bunch = pickle.load(elnino_fp)
    else:
        elnino_bunch = elnino.load()
        with open(elnino_pickle, 'wb') as elnino_fp:
            pickle.dump(elnino_bunch, elnino_fp)
    elnino_data = elnino_bunch['data']
    logger.info('elnino data has %d rows' % len(elnino_data))
    elnino_names = elnino_bunch['names']
    logger.info('elnino names: %s' % str(elnino_names))
    elnino_raw_data = elnino_bunch['raw_data']
    logger.info('elnino raw data is %d x %d' % elnino_raw_data.shape)

    logger.info('loading Engel food expenditure data')
    engel_pickle = data_folder + 'engel.pkl'
    if exists(engel_pickle):
        with open(engel_pickle, 'rb') as engel_fp:
            engel_bunch = pickle.load(engel_fp)
    else:
        engel_bunch = engel.load()
        with open(engel_pickle, 'wb') as engel_fp:
            pickle.dump(engel_bunch, engel_fp)
    engel_data = engel_bunch['data']
    logger.info('engel data has %d rows' % len(engel_data))
    engel_names = engel_bunch['names']
    logger.info('engel names: %s' % str(engel_names))
    engel_raw_data = engel_bunch['raw_data']
    logger.info('engel raw data is %d x %d' % engel_raw_data.shape)

    logger.info('loading extramarital affair data')
    fair_pickle = data_folder + 'fair.pkl'
    if exists(fair_pickle):
        with open(fair_pickle, 'rb') as fair_fp:
            fair_bunch = pickle.load(fair_fp)
    else:
        fair_bunch = fair.load()
        with open(fair_pickle, 'wb') as fair_fp:
            pickle.dump(fair_bunch, fair_fp)
    fair_data = fair_bunch['data']
    logger.info('fair data has %d rows' % len(fair_data))
    fair_names = fair_bunch['names']
    logger.info('fair names: %s' % str(fair_names))
    fair_raw_data = fair_bunch['raw_data']
    logger.info('fair raw data is %d x %d' % fair_raw_data.shape)

    logger.info('loading fertility data')
    fertility_pickle = data_folder + 'fertility.pkl'
    if exists(fertility_pickle):
        with open(fertility_pickle, 'rb') as fertility_fp:
            fertility_bunch = pickle.load(fertility_fp)
    else:
        fertility_bunch = fertility.load()
        with open(fertility_pickle, 'wb') as fertility_fp:
            pickle.dump(fertility_bunch, fertility_fp)
    fertility_data = fertility_bunch['data']
    logger.info('fertility data has %d rows' % len(fertility_data))
    fertility_names = fertility_bunch['names']
    logger.info('fertility names: %s' % str(fertility_names))

    logger.info('loading Grunfeld data')
    grunfeld_pickle = data_folder + 'grunfeld.pkl'
    if exists(grunfeld_pickle):
        with open(grunfeld_pickle, 'rb') as grunfeld_fp:
            grunfeld_bunch = pickle.load(grunfeld_fp)
    else:
        grunfeld_bunch = grunfeld.load()
        with open(grunfeld_pickle, 'wb') as grunfeld_fp:
            pickle.dump(grunfeld_bunch, grunfeld_fp)
    grunfeld_data = grunfeld_bunch['data']
    logger.info('grunfeld data has %d rows' % len(grunfeld_data))
    grunfeld_names = grunfeld_bunch['names']
    logger.info('grunfeld names: %s' % str(grunfeld_names))

    logger.info('loading heart transplant data')
    heart_pickle = data_folder + 'heart.pkl'
    if exists(heart_pickle):
        with open(heart_pickle, 'rb') as heart_fp:
            heart_bunch = pickle.load(heart_fp)
    else:
        heart_bunch = heart.load()
        with open(heart_pickle, 'wb') as heart_fp:
            pickle.dump(heart_bunch, heart_fp)
    heart_data = heart_bunch['data']
    logger.info('heart data has %d rows' % len(heart_data))
    heart_names = heart_bunch['names']
    logger.info('heart names: %s' % str(heart_names))

    logger.info('loading West German interest and inflation rate data')
    interest_inflation_pickle = data_folder + 'interest_inflation.pkl'
    if exists(interest_inflation_pickle):
        with open(interest_inflation_pickle, 'rb') as interest_inflation_fp:
            interest_inflation_bunch = pickle.load(interest_inflation_fp)
    else:
        interest_inflation_bunch = interest_inflation.load()
        with open(interest_inflation_pickle, 'wb') as interest_inflation_fp:
            pickle.dump(interest_inflation_bunch, interest_inflation_fp)
    interest_inflation_data = interest_inflation_bunch['data']
    logger.info('interest_inflation data has %d rows ' % len(interest_inflation_data))
    interest_inflation_names = interest_inflation_bunch['names']
    logger.info('interest_inflation names: %s' % str(interest_inflation_names))

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

    logger.info('loading Linnerud data')
    linnerud_bunch = load_linnerud(return_X_y=return_X_y)
    linnerud_data = linnerud_bunch['data']
    linnerud_feature_names = linnerud_bunch['feature_names']
    linnerud_target = linnerud_bunch['target']
    linnerud_target_names = linnerud_bunch['target_names']
    linnerud_description = linnerud_bunch['DESCR']

    logger.info('loading Longley macroeconomic data')
    longley_pickle = data_folder + 'longley.pkl'
    if exists(longley_pickle):
        with open(longley_pickle, 'rb') as longley_fp:
            longley_bunch = pickle.load(longley_fp)
    else:
        longley_bunch = longley.load()
        with open(longley_pickle, 'wb') as longley_fp:
            pickle.dump(longley_bunch, longley_fp)
    longley_data = longley_bunch['data']
    logger.info('longley data has %d rows ' % len(longley_data))
    longley_names = longley_bunch['names']
    logger.info('longley names: %s' % str(longley_names))

    logger.info('loading US macroeconomic data')
    macrodata_pickle = data_folder + 'macrodata.pkl'
    if exists(macrodata_pickle):
        with open(macrodata_pickle, 'rb') as macrodata_fp:
            macrodata_bunch = pickle.load(macrodata_fp)
    else:
        macrodata_bunch = macrodata.load()
        with open(macrodata_pickle, 'wb') as macrodata_fp:
            pickle.dump(macrodata_bunch, macrodata_fp)
    macrodata_data = macrodata_bunch['data']
    logger.info('macrodata data has %d rows ' % len(macrodata_data))
    macrodata_names = macrodata_bunch['names']
    logger.info('macrodata names: %s' % str(macrodata_names))

    logger.info('loading travel mode choice data')
    modechoice_pickle = data_folder + 'modechoice.pkl'
    if exists(modechoice_pickle):
        with open(modechoice_pickle, 'rb') as modechoice_fp:
            modechoice_bunch = pickle.load(modechoice_fp)
    else:
        modechoice_bunch = modechoice.load()
        with open(modechoice_pickle, 'wb') as modechoice_fp:
            pickle.dump(modechoice_bunch, modechoice_fp)
    modechoice_data = modechoice_bunch['data']
    logger.info('modechoice data has %d rows ' % len(modechoice_data))
    modechoice_names = modechoice_bunch['names']
    logger.info('modechoice names: %s' % str(modechoice_names))

    logger.info('loading newsgroups data')
    newsgroups_bunch = fetch_20newsgroups(data_home=data_folder)
    newsgroups_data = newsgroups_bunch['data']
    newsgroups_target_names = newsgroups_bunch['target_names']
    newsgroups_description = newsgroups_bunch['DESCR']
    newsgroups_target = newsgroups_bunch['target']
    newsgroups_filenames = newsgroups_bunch['filenames']

    logger.info('loading Nile river flows at Ashwan data')
    nile_pickle = data_folder + 'nile.pkl'
    if exists(nile_pickle):
        with open(nile_pickle, 'rb') as nile_fp:
            nile_bunch = pickle.load(nile_fp)
    else:
        nile_bunch = nile.load()
        with open(nile_pickle, 'wb') as nile_fp:
            pickle.dump(nile_bunch, nile_fp)
    nile_data = nile_bunch['data']
    logger.info('nile data has %d rows ' % len(nile_data))
    nile_names = nile_bunch['names']
    logger.info('nile names: %s' % str(nile_names))

    logger.info('loading Olivetti faces data')
    olivetti_faces = fetch_olivetti_faces(data_home=data_folder)
    olivetti_faces_data = olivetti_faces['data']
    olivetti_faces_images = olivetti_faces['images']
    olivetti_faces_target = olivetti_faces['target']
    olivetti_faces_description = olivetti_faces['DESCR']

    logger.info('loading RAND health insurance experiment data')
    randhie_pickle = data_folder + 'randhie.pkl'
    if exists(randhie_pickle):
        with open(randhie_pickle, 'rb') as randhie_fp:
            randhie_bunch = pickle.load(randhie_fp)
    else:
        randhie_bunch = randhie.load()
        with open(randhie_pickle, 'wb') as randhie_fp:
            pickle.dump(randhie_bunch, randhie_fp)
    randhie_data = randhie_bunch['data']
    logger.info('randhie data has %d rows ' % len(randhie_data))
    randhie_names = randhie_bunch['names']
    logger.info('randhie names: %s' % str(randhie_names))

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

    logger.info('loading Scotland taxation data')
    scotland_pickle = data_folder + 'scotland.pkl'
    if exists(scotland_pickle):
        with open(scotland_pickle, 'rb') as scotland_fp:
            scotland_bunch = pickle.load(scotland_fp)
    else:
        scotland_bunch = scotland.load()
        with open(scotland_pickle, 'wb') as scotland_fp:
            pickle.dump(scotland_bunch, scotland_fp)
    scotland_data = scotland_bunch['data']
    logger.info('scotland data has %d rows ' % len(scotland_data))
    scotland_names = scotland_bunch['names']
    logger.info('scotland names: %s' % str(scotland_names))

    logger.info('loading ships data')
    ships_pickle = data_folder + 'ships.pkl'
    if exists(ships_pickle):
        with open(ships_pickle, 'rb') as ships_fp:
            ships_bundle = pickle.load(ships_fp)
    else:
        ships_bundle = get_rdataset('ships', 'COUNT')
        with open(ships_pickle, 'wb') as ships_fp:
            pickle.dump(ships_bundle, ships_fp)
    ships_data = ships_bundle.data
    logger.info('ships data has variables %s and has %d rows' % (list(ships_data), len(ships_data)))
    ships_title = ships_bundle.title
    logger.info('ships data has title %s' % ships_title)

    logger.info('loading labor and income dynamics data')
    SLID_pickle = data_folder + 'SLID.pkl'
    if exists(SLID_pickle):
        with open(SLID_pickle, 'rb') as SLID_fp:
            SLID_bundle = pickle.load(SLID_fp)
    else:
        SLID_bundle = get_rdataset('SLID', 'carData')
        with open(SLID_pickle, 'wb') as SLID_fp:
            pickle.dump(SLID_bundle, SLID_fp)
    SLID_data = SLID_bundle.data
    logger.info('SLID data has variables %s and has %d rows' % (list(SLID_data), len(SLID_data)))
    SLID_title = SLID_bundle.title
    logger.info('SLID data has title %s' % SLID_title)

    logger.info('loading smoking data')
    smoking_pickle = data_folder + 'smoking.pkl'
    if exists(smoking_pickle):
        with open(smoking_pickle, 'rb') as smoking_fp:
            smoking_bundle = pickle.load(smoking_fp)
    else:
        smoking_bundle = get_rdataset('smoking', 'COUNT')
        with open(smoking_pickle, 'wb') as smoking_fp:
            pickle.dump(smoking_bundle, smoking_fp)
    smoking_data = smoking_bundle.data
    logger.info('smoking data has variables %s and has %d rows' % (list(smoking_data), len(smoking_data)))
    smoking_title = smoking_bundle.title
    logger.info('smoking data has title %s' % smoking_title)

    logger.info('loading soil composition data')
    Soils_pickle = data_folder + 'Soils.pkl'
    if exists(Soils_pickle):
        with open(Soils_pickle, 'rb') as Soils_fp:
            Soils_bundle = pickle.load(Soils_fp)
    else:
        Soils_bundle = get_rdataset('Soils', 'carData')
        with open(Soils_pickle, 'wb') as Soils_fp:
            pickle.dump(Soils_bundle, Soils_fp)
    Soils_data = Soils_bundle.data
    logger.info('Soils data has variables %s and has %d rows' % (list(Soils_data), len(Soils_data)))
    Soils_title = Soils_bundle.title
    logger.info('Soils data has title %s' % Soils_title)

    logger.info('loading Spector and Mazzeo program effectiveness data')
    spector_pickle = data_folder + 'spector.pkl'
    if exists(spector_pickle):
        with open(spector_pickle, 'rb') as spector_fp:
            spector_bunch = pickle.load(spector_fp)
    else:
        spector_bunch = spector.load()
        with open(spector_pickle, 'wb') as spector_fp:
            pickle.dump(spector_bunch, spector_fp)
    spector_data = spector_bunch['data']
    logger.info('spector data has %d rows ' % len(spector_data))
    spector_names = spector_bunch['names']
    logger.info('spector names: %s' % str(spector_names))

    logger.info('loading stack loss data')
    stackloss_pickle = data_folder + 'stackloss.pkl'
    if exists(stackloss_pickle):
        with open(stackloss_pickle, 'rb') as stackloss_fp:
            stackloss_bunch = pickle.load(stackloss_fp)
    else:
        stackloss_bunch = stackloss.load()
        with open(stackloss_pickle, 'wb') as stackloss_fp:
            pickle.dump(stackloss_bunch, stackloss_fp)
    stackloss_data = stackloss_bunch['data']
    logger.info('stackloss data has %d rows ' % len(stackloss_data))
    stackloss_names = stackloss_bunch['names']
    logger.info('stackloss names: %s' % str(stackloss_names))

    logger.info('loading Star98 educational data')
    star98_pickle = data_folder + 'star98.pkl'
    if exists(star98_pickle):
        with open(star98_pickle, 'rb') as star98_fp:
            star98_bunch = pickle.load(star98_fp)
    else:
        star98_bunch = star98.load()
        with open(star98_pickle, 'wb') as star98_fp:
            pickle.dump(star98_bunch, star98_fp)
    star98_data = star98_bunch['data']
    logger.info('star98 data has %d rows ' % len(star98_data))
    star98_names = star98_bunch['names']
    logger.info('star98 names: %s' % str(star98_names))

    logger.info('loading State Crime data')
    statecrime_pickle = data_folder + 'statecrime.pkl'
    if exists(statecrime_pickle):
        with open(statecrime_pickle, 'rb') as statecrime_fp:
            statecrime_bunch = pickle.load(statecrime_fp)
    else:
        statecrime_bunch = statecrime.load()
        with open(statecrime_pickle, 'wb') as statecrime_fp:
            pickle.dump(statecrime_bunch, statecrime_fp)
    statecrime_data = statecrime_bunch['data']
    logger.info('statecrime data has %d rows ' % len(statecrime_data))
    statecrime_names = statecrime_bunch['names']
    logger.info('statecrime names: %s' % str(statecrime_names))

    logger.info('loading education and related statistics data')
    States_pickle = data_folder + 'States.pkl'
    if exists(States_pickle):
        with open(States_pickle, 'rb') as States_fp:
            States_bundle = pickle.load(States_fp)
    else:
        States_bundle = get_rdataset('States', 'carData')
        with open(States_pickle, 'wb') as States_fp:
            pickle.dump(States_bundle, States_fp)
    States_data = States_bundle.data
    logger.info('States data has variables %s and has %d rows' % (list(States_data), len(States_data)))
    States_title = States_bundle.title
    logger.info('States data has title %s' % States_title)

    logger.info('loading US strike duration data')
    strikes_pickle = data_folder + 'strikes.pkl'
    if exists(strikes_pickle):
        with open(strikes_pickle, 'rb') as strikes_fp:
            strikes_bunch = pickle.load(strikes_fp)
    else:
        strikes_bunch = strikes.load()
        with open(strikes_pickle, 'wb') as strikes_fp:
            pickle.dump(strikes_bunch, strikes_fp)
    strikes_data = strikes_bunch['data']
    logger.info('strikes data has %d rows ' % len(strikes_data))
    strikes_names = strikes_bunch['names']
    logger.info('strikes names: %s' % str(strikes_names))

    logger.info('loading yearly sunspot data')
    sunspots_pickle = data_folder + 'sunspots.pkl'
    if exists(sunspots_pickle):
        with open(sunspots_pickle, 'rb') as sunspots_fp:
            sunspots_bunch = pickle.load(sunspots_fp)
    else:
        sunspots_bunch = sunspots.load()
        with open(sunspots_pickle, 'wb') as sunspots_fp:
            pickle.dump(sunspots_bunch, sunspots_fp)
    sunspots_data = sunspots_bunch['data']
    logger.info('sunspots data has %d rows ' % len(sunspots_data))
    sunspots_names = sunspots_bunch['names']
    logger.info('sunspots names: %s' % str(sunspots_names))

    logger.info('loading cancer survival data')
    survival_pickle = data_folder + 'survival.pkl'
    if exists(survival_pickle):
        with open(survival_pickle, 'rb') as survival_fp:
            survival_bundle = pickle.load(survival_fp)
    else:
        survival_bundle = get_rdataset('survival', 'boot')
        with open(survival_pickle, 'wb') as survival_fp:
            pickle.dump(survival_bundle, survival_fp)
    survival_data = survival_bundle.data
    logger.info('survival data has variables %s and has %d rows' % (list(survival_data), len(survival_data)))
    survival_title = survival_bundle.title
    logger.info('survival data has title %s' % survival_title)

    logger.info('loading Tau particle data')
    tau_pickle = data_folder + 'tau.pkl'
    if exists(tau_pickle):
        with open(tau_pickle, 'rb') as tau_fp:
            tau_bundle = pickle.load(tau_fp)
    else:
        tau_bundle = get_rdataset('tau', 'boot')
        with open(tau_pickle, 'wb') as tau_fp:
            pickle.dump(tau_bundle, tau_fp)
    tau_data = tau_bundle.data
    logger.info('tau data has variables %s and has %d rows' % (list(tau_data), len(tau_data)))
    tau_title = tau_bundle.title
    logger.info('tau data has title %s' % tau_title)

    logger.info('loading titanic data')
    titanic_pickle = data_folder + 'titanic.pkl'
    if exists(titanic_pickle):
        with open(titanic_pickle, 'rb') as titanic_fp:
            titanic_bundle = pickle.load(titanic_fp)
    else:
        titanic_bundle = get_rdataset('titanic', 'COUNT')
        with open(titanic_pickle, 'wb') as titanic_fp:
            pickle.dump(titanic_bundle, titanic_fp)
    titanic_data = titanic_bundle.data
    logger.info('titanic data has variables %s and has %d rows' % (list(titanic_data), len(titanic_data)))
    titanic_title = titanic_bundle.title
    logger.info('titanic data has title %s' % titanic_title)

    logger.info('loading titanicgrp data')
    titanicgrp_pickle = data_folder + 'titanicgrp.pkl'
    if exists(titanicgrp_pickle):
        with open(titanicgrp_pickle, 'rb') as titanicgrp_fp:
            titanicgrp_bundle = pickle.load(titanicgrp_fp)
    else:
        titanicgrp_bundle = get_rdataset('titanicgrp', 'COUNT')
        with open(titanicgrp_pickle, 'wb') as titanicgrp_fp:
            pickle.dump(titanicgrp_bundle, titanicgrp_fp)
    titanicgrp_data = titanicgrp_bundle.data
    logger.info('titanicgrp data has variables %s and has %d rows' % (list(titanicgrp_data), len(titanicgrp_data)))
    titanicgrp_title = titanicgrp_bundle.title
    logger.info('titanicgrp data has title %s' % titanicgrp_title)

    logger.info('loading HMS Titanic survival data')
    TitanicSurvival_pickle = data_folder + 'TitanicSurvival.pkl'
    if exists(TitanicSurvival_pickle):
        with open(TitanicSurvival_pickle, 'rb') as TitanicSurvival_fp:
            TitanicSurvival_bundle = pickle.load(TitanicSurvival_fp)
    else:
        TitanicSurvival_bundle = get_rdataset('TitanicSurvival', 'carData')
        with open(TitanicSurvival_pickle, 'wb') as TitanicSurvival_fp:
            pickle.dump(TitanicSurvival_bundle, TitanicSurvival_fp)
    TitanicSurvival_data = TitanicSurvival_bundle.data
    logger.info('TitanicSurvival data has variables %s and has %d rows' % (
        list(TitanicSurvival_data), len(TitanicSurvival_data)))
    TitanicSurvival_title = TitanicSurvival_bundle.title
    logger.info('TitanicSurvival data has title %s' % TitanicSurvival_title)

    logger.info('loading transaction data')
    Transact_pickle = data_folder + 'Transact.pkl'
    if exists(Transact_pickle):
        with open(Transact_pickle, 'rb') as Transact_fp:
            Transact_bundle = pickle.load(Transact_fp)
    else:
        Transact_bundle = get_rdataset('Transact', 'carData')
        with open(Transact_pickle, 'wb') as Transact_fp:
            pickle.dump(Transact_bundle, Transact_fp)
    Transact_data = Transact_bundle.data
    logger.info('Transact data has variables %s and has %d rows' % (
        list(Transact_data), len(Transact_data)))
    Transact_title = Transact_bundle.title
    logger.info('Transact data has title %s' % Transact_title)

    logger.info('loading tuna sighting data')
    tuna_pickle = data_folder + 'tuna.pkl'
    if exists(tuna_pickle):
        with open(tuna_pickle, 'rb') as tuna_fp:
            tuna_bundle = pickle.load(tuna_fp)
    else:
        tuna_bundle = get_rdataset('tuna', 'boot')
        with open(tuna_pickle, 'wb') as tuna_fp:
            pickle.dump(tuna_bundle, tuna_fp)
    tuna_data = tuna_bundle.data
    logger.info('tuna data has variables %s and has %d rows' % (list(tuna_data), len(tuna_data)))
    tuna_title = tuna_bundle.title
    logger.info('tuna data has title %s' % tuna_title)

    logger.info('loading UN national statistics data')
    UN_pickle = data_folder + 'UN.pkl'
    if exists(UN_pickle):
        with open(UN_pickle, 'rb') as UN_fp:
            UN_bundle = pickle.load(UN_fp)
    else:
        UN_bundle = get_rdataset('UN', 'carData')
        with open(UN_pickle, 'wb') as UN_fp:
            pickle.dump(UN_bundle, UN_fp)
    UN_data = UN_bundle.data
    logger.info('UN data has variables %s and has %d rows' % (
        list(UN_data), len(UN_data)))
    UN_title = UN_bundle.title
    logger.info('UN data has title %s' % UN_title)

    logger.info('loading urine analysis data')
    urine_pickle = data_folder + 'urine.pkl'
    if exists(urine_pickle):
        with open(urine_pickle, 'rb') as urine_fp:
            urine_bundle = pickle.load(urine_fp)
    else:
        urine_bundle = get_rdataset('urine', 'boot')
        with open(urine_pickle, 'wb') as urine_fp:
            pickle.dump(urine_bundle, urine_fp)
    urine_data = urine_bundle.data
    logger.info('urine data has variables %s and has %d rows' % (list(urine_data), len(urine_data)))
    urine_title = urine_bundle.title
    logger.info('urine data has title %s' % urine_title)

    logger.info('loading US population data')
    USPop_pickle = data_folder + 'USPop.pkl'
    if exists(USPop_pickle):
        with open(USPop_pickle, 'rb') as USPop_fp:
            USPop_bundle = pickle.load(USPop_fp)
    else:
        USPop_bundle = get_rdataset('USPop', 'carData')
        with open(USPop_pickle, 'wb') as USPop_fp:
            pickle.dump(USPop_bundle, USPop_fp)
    USPop_data = USPop_bundle.data
    logger.info('USPop data has variables %s and has %d rows' % (
        list(USPop_data), len(USPop_data)))
    USPop_title = USPop_bundle.title
    logger.info('USPop data has title %s' % USPop_title)

    logger.info('loading vocabulary and education data')
    Vocab_pickle = data_folder + 'Vocab.pkl'
    if exists(Vocab_pickle):
        with open(Vocab_pickle, 'rb') as Vocab_fp:
            Vocab_bundle = pickle.load(Vocab_fp)
    else:
        Vocab_bundle = get_rdataset('Vocab', 'carData')
        with open(Vocab_pickle, 'wb') as Vocab_fp:
            pickle.dump(Vocab_bundle, Vocab_fp)
    Vocab_data = Vocab_bundle.data
    logger.info('Vocab data has variables %s and has %d rows' % (
        list(Vocab_data), len(Vocab_data)))
    Vocab_title = Vocab_bundle.title
    logger.info('Vocab data has title %s' % Vocab_title)

    logger.info('loading Presidential Republican voting dat data')
    votes_repub_pickle = data_folder + 'votes.repub.pkl'
    if exists(votes_repub_pickle):
        with open(votes_repub_pickle, 'rb') as votes_repub_fp:
            votes_repub_bundle = pickle.load(votes_repub_fp)
    else:
        votes_repub_bundle = get_rdataset('votes.repub', 'cluster')
        with open(votes_repub_pickle, 'wb') as votes_repub_fp:
            pickle.dump(votes_repub_bundle, votes_repub_fp)
    votes_repub_data = votes_repub_bundle.data
    logger.info('votes.repub data has variables %s and has %d rows' % (list(votes_repub_data), len(votes_repub_data)))
    votes_repub_title = votes_repub_bundle.title
    logger.info('votes.repub data has title %s' % votes_repub_title)

    logger.info('loading weight loss data')
    WeightLoss_pickle = data_folder + 'WeightLoss.pkl'
    if exists(WeightLoss_pickle):
        with open(WeightLoss_pickle, 'rb') as WeightLoss_fp:
            WeightLoss_bundle = pickle.load(WeightLoss_fp)
    else:
        WeightLoss_bundle = get_rdataset('WeightLoss', 'carData')
        with open(WeightLoss_pickle, 'wb') as WeightLoss_fp:
            pickle.dump(WeightLoss_bundle, WeightLoss_fp)
    WeightLoss_data = WeightLoss_bundle.data
    logger.info('WeightLoss data has variables %s and has %d rows' % (
        list(WeightLoss_data), len(WeightLoss_data)))
    WeightLoss_title = WeightLoss_bundle.title
    logger.info('WeightLoss data has title %s' % WeightLoss_title)

    logger.info('loading Bangladeshi well data')
    Wells_pickle = data_folder + 'Wells.pkl'
    if exists(Wells_pickle):
        with open(Wells_pickle, 'rb') as Wells_fp:
            Wells_bundle = pickle.load(Wells_fp)
    else:
        Wells_bundle = get_rdataset('Wells', 'carData')
        with open(Wells_pickle, 'wb') as Wells_fp:
            pickle.dump(Wells_bundle, Wells_fp)
    Wells_data = Wells_bundle.data
    logger.info('Wells data has variables %s and has %d rows' % (
        list(Wells_data), len(Wells_data)))
    Wells_title = Wells_bundle.title
    logger.info('Wells data has title %s' % Wells_title)

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

    logger.info('loading Women\'s labor force participation data')
    Womenlf_pickle = data_folder + 'Womenlf.pkl'
    if exists(Womenlf_pickle):
        with open(Womenlf_pickle, 'rb') as Womenlf_fp:
            Womenlf_bundle = pickle.load(Womenlf_fp)
    else:
        Womenlf_bundle = get_rdataset('Womenlf', 'carData')
        with open(Womenlf_pickle, 'wb') as Womenlf_fp:
            pickle.dump(Womenlf_bundle, Womenlf_fp)
    Womenlf_data = Womenlf_bundle.data
    logger.info('Womenlf data has variables %s and has %d rows' % (
        list(Womenlf_data), len(Womenlf_data)))
    Womenlf_title = Womenlf_bundle.title
    logger.info('Womenlf data has title %s' % Womenlf_title)

    logger.info('loading post-coma IQ recovery data')
    Wong_pickle = data_folder + 'Wong.pkl'
    if exists(Wong_pickle):
        with open(Wong_pickle, 'rb') as Wong_fp:
            Wong_bundle = pickle.load(Wong_fp)
    else:
        Wong_bundle = get_rdataset('Wong', 'carData')
        with open(Wong_pickle, 'wb') as Wong_fp:
            pickle.dump(Wong_bundle, Wong_fp)
    Wong_data = Wong_bundle.data
    logger.info('Wong data has variables %s and has %d rows' % (
        list(Wong_data), len(Wong_data)))
    Wong_title = Wong_bundle.title
    logger.info('Wong data has title %s' % Wong_title)

    logger.info('loading wool data')
    Wool_pickle = data_folder + 'Wool.pkl'
    if exists(Wool_pickle):
        with open(Wool_pickle, 'rb') as Wool_fp:
            Wool_bundle = pickle.load(Wool_fp)
    else:
        Wool_bundle = get_rdataset('Wool', 'carData')
        with open(Wool_pickle, 'wb') as Wool_fp:
            pickle.dump(Wool_bundle, Wool_fp)
    Wool_data = Wool_bundle.data
    logger.info('Wool data has variables %s and has %d rows' % (
        list(Wool_data), len(Wool_data)))
    Wool_title = Wool_bundle.title
    logger.info('Wool data has title %s' % Wool_title)

    logger.info('loading Australian wool data')
    wool_pickle = data_folder + 'wool.pkl'
    if exists(wool_pickle):
        with open(wool_pickle, 'rb') as wool_fp:
            wool_bundle = pickle.load(wool_fp)
    else:
        wool_bundle = get_rdataset('wool', 'boot')
        with open(wool_pickle, 'wb') as wool_fp:
            pickle.dump(wool_bundle, wool_fp)
    wool_data = wool_bundle.data
    logger.info('wool data has variables %s and has %d rows' % (list(wool_data), len(wool_data)))
    wool_title = wool_bundle.title
    logger.info('wool data has title %s' % wool_title)

    logger.info('loading World Values Survey data')
    WVS_pickle = data_folder + 'WVS.pkl'
    if exists(WVS_pickle):
        with open(WVS_pickle, 'rb') as WVS_fp:
            WVS_bundle = pickle.load(WVS_fp)
    else:
        WVS_bundle = get_rdataset('WVS', 'carData')
        with open(WVS_pickle, 'wb') as WVS_fp:
            pickle.dump(WVS_bundle, WVS_fp)
    WVS_data = WVS_bundle.data
    logger.info('WVS data has variables %s and has %d rows' % (
        list(WVS_data), len(WVS_data)))
    WVS_title = WVS_bundle.title
    logger.info('WVS data has title %s' % WVS_title)

    logger.info('loading bivariate three-cluster data')
    xclara_pickle = data_folder + 'xclara.pkl'
    if exists(xclara_pickle):
        with open(xclara_pickle, 'rb') as xclara_fp:
            xclara_bundle = pickle.load(xclara_fp)
    else:
        xclara_bundle = get_rdataset('xclara', 'cluster')
        with open(xclara_pickle, 'wb') as xclara_fp:
            pickle.dump(xclara_bundle, xclara_fp)
    xclara_data = xclara_bundle.data
    logger.info('xclara data has variables %s and has %d rows' % (list(xclara_data), len(xclara_data)))
    xclara_title = xclara_bundle.title
    logger.info('xclara data has title %s' % xclara_title)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
