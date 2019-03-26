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

    logger.info('loading frogs data')
    frogs_pickle = data_folder + 'frogs.pkl'
    if exists(frogs_pickle):
        with open(frogs_pickle, 'rb') as frogs_fp:
            frogs_bundle = pickle.load(frogs_fp)
    else:
        frogs_bundle = get_rdataset('frogs', 'DAAG')
        with open(frogs_pickle, 'wb') as frogs_fp:
            pickle.dump(frogs_bundle, frogs_fp)
    frogs_data = frogs_bundle.data
    logger.info('frogs data has variables %s' % list(frogs_data))
    logger.info('frogs data has %d rows and %d variables' % frogs_data.shape)
    frogs_title = frogs_bundle.title
    logger.info('frogs data has title %s' % frogs_title)

    logger.info('loading Frosted Flakes data')
    frostedflakes_pickle = data_folder + 'frostedflakes.pkl'
    if exists(frostedflakes_pickle):
        with open(frostedflakes_pickle, 'rb') as frostedflakes_fp:
            frostedflakes_bundle = pickle.load(frostedflakes_fp)
    else:
        frostedflakes_bundle = get_rdataset('frostedflakes', 'DAAG')
        with open(frostedflakes_pickle, 'wb') as frostedflakes_fp:
            pickle.dump(frostedflakes_bundle, frostedflakes_fp)
    frostedflakes_data = frostedflakes_bundle.data
    logger.info('frostedflakes data has variables %s' % list(frostedflakes_data))
    logger.info('frostedflakes data has %d rows and %d variables' % frostedflakes_data.shape)
    frostedflakes_title = frostedflakes_bundle.title
    logger.info('frostedflakes data has title %s' % frostedflakes_title)

    logger.info('loading electrical resistance of kiwifruit data')
    fruitohms_pickle = data_folder + 'fruitohms.pkl'
    if exists(fruitohms_pickle):
        with open(fruitohms_pickle, 'rb') as fruitohms_fp:
            fruitohms_bundle = pickle.load(fruitohms_fp)
    else:
        fruitohms_bundle = get_rdataset('fruitohms', 'DAAG')
        with open(fruitohms_pickle, 'wb') as fruitohms_fp:
            pickle.dump(fruitohms_bundle, fruitohms_fp)
    fruitohms_data = fruitohms_bundle.data
    logger.info('fruitohms data has variables %s' % list(fruitohms_data))
    logger.info('fruitohms data has %d rows and %d variables' % fruitohms_data.shape)
    fruitohms_title = fruitohms_bundle.title
    logger.info('fruitohms data has title %s' % fruitohms_title)

    logger.info('loading pentazocine effect data')
    gaba_pickle = data_folder + 'gaba.pkl'
    if exists(gaba_pickle):
        with open(gaba_pickle, 'rb') as gaba_fp:
            gaba_bundle = pickle.load(gaba_fp)
    else:
        gaba_bundle = get_rdataset('gaba', 'DAAG')
        with open(gaba_pickle, 'wb') as gaba_fp:
            pickle.dump(gaba_bundle, gaba_fp)
    gaba_data = gaba_bundle.data
    logger.info('gaba data has variables %s' % list(gaba_data))
    logger.info('gaba data has %d rows and %d variables' % gaba_data.shape)
    gaba_title = gaba_bundle.title
    logger.info('gaba data has title %s' % gaba_title)

    logger.info('loading seismic timing data')
    geophones_pickle = data_folder + 'geophones.pkl'
    if exists(geophones_pickle):
        with open(geophones_pickle, 'rb') as geophones_fp:
            geophones_bundle = pickle.load(geophones_fp)
    else:
        geophones_bundle = get_rdataset('geophones', 'DAAG')
        with open(geophones_pickle, 'wb') as geophones_fp:
            pickle.dump(geophones_bundle, geophones_fp)
    geophones_data = geophones_bundle.data
    logger.info('geophones data has variables %s' % list(geophones_data))
    logger.info('geophones data has %d rows and %d variables' % geophones_data.shape)
    geophones_title = geophones_bundle.title
    logger.info('geophones data has title %s' % geophones_title)

    logger.info('loading depression data')
    Ginzberg_pickle = data_folder + 'Ginzberg.pkl'
    if exists(Ginzberg_pickle):
        with open(Ginzberg_pickle, 'rb') as Ginzberg_fp:
            Ginzberg_bundle = pickle.load(Ginzberg_fp)
    else:
        Ginzberg_bundle = get_rdataset('Ginzberg', 'carData')
        with open(Ginzberg_pickle, 'wb') as Ginzberg_fp:
            pickle.dump(Ginzberg_bundle, Ginzberg_fp)
    Ginzberg_data = Ginzberg_bundle.data
    logger.info(
        'Ginzberg data has variables %s and has %d rows' % (list(Ginzberg_data), len(Ginzberg_data)))
    Ginzberg_title = Ginzberg_bundle.title
    logger.info('Ginzberg data has title %s' % Ginzberg_title)

    logger.info('loading acceleration due to gravity data')
    grav_pickle = data_folder + 'grav.pkl'
    if exists(grav_pickle):
        with open(grav_pickle, 'rb') as grav_fp:
            grav_bundle = pickle.load(grav_fp)
    else:
        grav_bundle = get_rdataset('grav', 'boot')
        with open(grav_pickle, 'wb') as grav_fp:
            pickle.dump(grav_bundle, grav_fp)
    grav_data = grav_bundle.data
    logger.info('grav data has variables %s and has %d rows' % (list(grav_data), len(grav_data)))
    grav_title = grav_bundle.title
    logger.info('grav data has title %s' % grav_title)

    logger.info('loading acceleration due to gravity data')
    gravity_pickle = data_folder + 'gravity.pkl'
    if exists(gravity_pickle):
        with open(gravity_pickle, 'rb') as gravity_fp:
            gravity_bundle = pickle.load(gravity_fp)
    else:
        gravity_bundle = get_rdataset('gravity', 'boot')
        with open(gravity_pickle, 'wb') as gravity_fp:
            pickle.dump(gravity_bundle, gravity_fp)
    gravity_data = gravity_bundle.data
    logger.info('gravity data has variables %s and has %d rows' % (list(gravity_data), len(gravity_data)))
    gravity_title = gravity_bundle.title
    logger.info('gravity data has title %s' % gravity_title)

    logger.info('loading great lakes yearly average height data')
    greatLakes_pickle = data_folder + 'greatLakes.pkl'
    if exists(greatLakes_pickle):
        with open(greatLakes_pickle, 'rb') as greatLakes_fp:
            greatLakes_bundle = pickle.load(greatLakes_fp)
    else:
        greatLakes_bundle = get_rdataset('greatLakes', 'DAAG')
        with open(greatLakes_pickle, 'wb') as greatLakes_fp:
            pickle.dump(greatLakes_bundle, greatLakes_fp)
    greatLakes_data = greatLakes_bundle.data
    logger.info('greatLakes data has variables %s' % list(greatLakes_data))
    logger.info('greatLakes data has %d rows and %d variables' % greatLakes_data.shape)
    greatLakes_title = greatLakes_bundle.title
    logger.info('greatLakes data has title %s' % greatLakes_title)

    logger.info('loading AUS/NZ alcohol consumption data')
    grog_pickle = data_folder + 'grog.pkl'
    if exists(grog_pickle):
        with open(grog_pickle, 'rb') as grog_fp:
            grog_bundle = pickle.load(grog_fp)
    else:
        grog_bundle = get_rdataset('grog', 'DAAG')
        with open(grog_pickle, 'wb') as grog_fp:
            pickle.dump(grog_bundle, grog_fp)
    grog_data = grog_bundle.data
    logger.info('grog data has variables %s' % list(grog_data))
    logger.info('grog data has %d rows and %d variables' % grog_data.shape)
    grog_title = grog_bundle.title
    logger.info('grog data has title %s' % grog_title)

    logger.info('loading refugee appeals data')
    Greene_pickle = data_folder + 'Greene.pkl'
    if exists(Greene_pickle):
        with open(Greene_pickle, 'rb') as Greene_fp:
            Greene_bundle = pickle.load(Greene_fp)
    else:
        Greene_bundle = get_rdataset('Greene', 'carData')
        with open(Greene_pickle, 'wb') as Greene_fp:
            pickle.dump(Greene_bundle, Greene_fp)
    Greene_data = Greene_bundle.data
    logger.info(
        'Greene data has variables %s and has %d rows' % (list(Greene_data), len(Greene_data)))
    Greene_title = Greene_bundle.title
    logger.info('Greene data has title %s' % Greene_title)

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

    logger.info('loading general social survey data')
    GSSvocab_pickle = data_folder + 'GSSvocab.pkl'
    if exists(GSSvocab_pickle):
        with open(GSSvocab_pickle, 'rb') as GSSvocab_fp:
            GSSvocab_bundle = pickle.load(GSSvocab_fp)
    else:
        GSSvocab_bundle = get_rdataset('GSSvocab', 'carData')
        with open(GSSvocab_pickle, 'wb') as GSSvocab_fp:
            pickle.dump(GSSvocab_bundle, GSSvocab_fp)
    GSSvocab_data = GSSvocab_bundle.data
    logger.info(
        'GSSvocab data has variables %s and has %d rows' % (list(GSSvocab_data), len(GSSvocab_data)))
    GSSvocab_title = GSSvocab_bundle.title
    logger.info('GSSvocab data has title %s' % GSSvocab_title)

    logger.info('loading anonymity/cooperation data')
    Guyer_pickle = data_folder + 'Guyer.pkl'
    if exists(Guyer_pickle):
        with open(Guyer_pickle, 'rb') as Guyer_fp:
            Guyer_bundle = pickle.load(Guyer_fp)
    else:
        Guyer_bundle = get_rdataset('Guyer', 'carData')
        with open(Guyer_pickle, 'wb') as Guyer_fp:
            pickle.dump(Guyer_bundle, Guyer_fp)
    Guyer_data = Guyer_bundle.data
    logger.info(
        'Guyer data has variables %s and has %d rows' % (list(Guyer_data), len(Guyer_data)))
    Guyer_title = Guyer_bundle.title
    logger.info('Guyer data has title %s' % Guyer_title)

    logger.info('loading Canadian crime data')
    Hartnagel_pickle = data_folder + 'Hartnagel.pkl'
    if exists(Hartnagel_pickle):
        with open(Hartnagel_pickle, 'rb') as Hartnagel_fp:
            Hartnagel_bundle = pickle.load(Hartnagel_fp)
    else:
        Hartnagel_bundle = get_rdataset('Hartnagel', 'carData')
        with open(Hartnagel_pickle, 'wb') as Hartnagel_fp:
            pickle.dump(Hartnagel_bundle, Hartnagel_fp)
    Hartnagel_data = Hartnagel_bundle.data
    logger.info(
        'Hartnagel data has variables %s and has %d rows' % (list(Hartnagel_data), len(Hartnagel_data)))
    Hartnagel_title = Hartnagel_bundle.title
    logger.info('Hartnagel data has title %s' % Hartnagel_title)

    logger.info('loading simulated minor head injury data')
    headinjury1_pickle = data_folder + 'headinjury1.pkl'
    if exists(headinjury1_pickle):
        with open(headinjury1_pickle, 'rb') as headinjury1_fp:
            headinjury1_bundle = pickle.load(headinjury1_fp)
    else:
        headinjury1_bundle = get_rdataset('headInjury', 'DAAG')
        with open(headinjury1_pickle, 'wb') as headinjury1_fp:
            pickle.dump(headinjury1_bundle, headinjury1_fp)
    headinjury1_data = headinjury1_bundle.data
    logger.info('headinjury1 data has variables %s' % list(headinjury1_data))
    logger.info('headinjury1 data has %d rows and %d variables' % headinjury1_data.shape)
    headinjury1_title = headinjury1_bundle.title
    logger.info('headinjury1 data has title %s' % headinjury1_title)

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

    logger.info('loading highway accident data')
    Highway1_pickle = data_folder + 'Highway1.pkl'
    if exists(Highway1_pickle):
        with open(Highway1_pickle, 'rb') as Highway1_fp:
            Highway1_bundle = pickle.load(Highway1_fp)
    else:
        Highway1_bundle = get_rdataset('Highway1', 'carData')
        with open(Highway1_pickle, 'wb') as Highway1_fp:
            pickle.dump(Highway1_bundle, Highway1_fp)
    Highway1_data = Highway1_bundle.data
    logger.info(
        'Highway1 data has variables %s and has %d rows' % (list(Highway1_data), len(Highway1_data)))
    Highway1_title = Highway1_bundle.title
    logger.info('Highway1 data has title %s' % Highway1_title)

    logger.info('loading Scottish hill races data')
    hills_pickle = data_folder + 'hills.pkl'
    if exists(hills_pickle):
        with open(hills_pickle, 'rb') as hills_fp:
            hills_bundle = pickle.load(hills_fp)
    else:
        hills_bundle = get_rdataset('hills', 'DAAG')
        with open(hills_pickle, 'wb') as hills_fp:
            pickle.dump(hills_bundle, hills_fp)
    hills_data = hills_bundle.data
    logger.info('hills data has variables %s' % list(hills_data))
    logger.info('hills data has %d rows and %d variables' % hills_data.shape)
    hills_title = hills_bundle.title
    logger.info('hills data has title %s' % hills_title)

    logger.info('loading Scottish hill races (2000) data')
    hills2000_pickle = data_folder + 'hills2000.pkl'
    if exists(hills2000_pickle):
        with open(hills2000_pickle, 'rb') as hills2000_fp:
            hills2000_bundle = pickle.load(hills2000_fp)
    else:
        hills2000_bundle = get_rdataset('hills2000', 'DAAG')
        with open(hills2000_pickle, 'wb') as hills2000_fp:
            pickle.dump(hills2000_bundle, hills2000_fp)
    hills2000_data = hills2000_bundle.data
    logger.info('hills2000 data has variables %s' % list(hills2000_data))
    logger.info('hills2000 data has %d rows and %d variables' % hills2000_data.shape)
    hills2000_title = hills2000_bundle.title
    logger.info('hills2000 data has title %s' % hills2000_title)

    logger.info('loading failure time of PET film data')
    hirose_pickle = data_folder + 'hirose.pkl'
    if exists(hirose_pickle):
        with open(hirose_pickle, 'rb') as hirose_fp:
            hirose_bundle = pickle.load(hirose_fp)
    else:
        hirose_bundle = get_rdataset('hirose', 'boot')
        with open(hirose_pickle, 'wb') as hirose_fp:
            pickle.dump(hirose_bundle, hirose_fp)
    hirose_data = hirose_bundle.data
    logger.info('hirose data has variables %s and has %d rows' % (list(hirose_data), len(hirose_data)))
    hirose_title = hirose_bundle.title
    logger.info('hirose data has title %s' % hirose_title)

    logger.info('loading Hawaiian Potassium-Argon age data')
    hotspots_pickle = data_folder + 'hotspots.pkl'
    if exists(hotspots_pickle):
        with open(hotspots_pickle, 'rb') as hotspots_fp:
            hotspots_bundle = pickle.load(hotspots_fp)
    else:
        hotspots_bundle = get_rdataset('hotspots', 'DAAG')
        with open(hotspots_pickle, 'wb') as hotspots_fp:
            pickle.dump(hotspots_bundle, hotspots_fp)
    hotspots_data = hotspots_bundle.data
    logger.info('hotspots data has variables %s' % list(hotspots_data))
    logger.info('hotspots data has %d rows and %d variables' % hotspots_data.shape)
    hotspots_title = hotspots_bundle.title
    logger.info('hotspots data has title %s' % hotspots_title)

    logger.info('loading Hawaiian Argon-Argon age data')
    hotspots2006_pickle = data_folder + 'hotspots2006.pkl'
    if exists(hotspots2006_pickle):
        with open(hotspots2006_pickle, 'rb') as hotspots2006_fp:
            hotspots2006_bundle = pickle.load(hotspots2006_fp)
    else:
        hotspots2006_bundle = get_rdataset('hotspots2006', 'DAAG')
        with open(hotspots2006_pickle, 'wb') as hotspots2006_fp:
            pickle.dump(hotspots2006_bundle, hotspots2006_fp)
    hotspots2006_data = hotspots2006_bundle.data
    logger.info('hotspots2006 data has variables %s' % list(hotspots2006_data))
    logger.info('hotspots2006 data has %d rows and %d variables' % hotspots2006_data.shape)
    hotspots2006_title = hotspots2006_bundle.title
    logger.info('hotspots2006 data has title %s' % hotspots2006_title)

    logger.info('loading Aranda house price data')
    houseprices_pickle = data_folder + 'houseprices.pkl'
    if exists(houseprices_pickle):
        with open(houseprices_pickle, 'rb') as houseprices_fp:
            houseprices_bundle = pickle.load(houseprices_fp)
    else:
        houseprices_bundle = get_rdataset('houseprices', 'DAAG')
        with open(houseprices_pickle, 'wb') as houseprices_fp:
            pickle.dump(houseprices_bundle, houseprices_fp)
    houseprices_data = houseprices_bundle.data
    logger.info('houseprices data has variables %s' % list(houseprices_data))
    logger.info('houseprices data has %d rows and %d variables' % houseprices_data.shape)
    houseprices_title = houseprices_bundle.title
    logger.info('houseprices data has title %s' % houseprices_title)

    logger.info('loading Oxygen uptake vs power output part 1 data')
    humanpower1_pickle = data_folder + 'humanpower1.pkl'
    if exists(humanpower1_pickle):
        with open(humanpower1_pickle, 'rb') as humanpower1_fp:
            humanpower1_bundle = pickle.load(humanpower1_fp)
    else:
        humanpower1_bundle = get_rdataset('humanpower1', 'DAAG')
        with open(humanpower1_pickle, 'wb') as humanpower1_fp:
            pickle.dump(humanpower1_bundle, humanpower1_fp)
    humanpower1_data = humanpower1_bundle.data
    logger.info('humanpower1 data has variables %s' % list(humanpower1_data))
    logger.info('humanpower1 data has %d rows and %d variables' % humanpower1_data.shape)
    humanpower1_title = humanpower1_bundle.title
    logger.info('humanpower1 data has title %s' % humanpower1_title)

    logger.info('loading named US Atlantic storm data')
    hurricNamed_pickle = data_folder + 'hurricNamed.pkl'
    if exists(hurricNamed_pickle):
        with open(hurricNamed_pickle, 'rb') as hurricNamed_fp:
            hurricNamed_bundle = pickle.load(hurricNamed_fp)
    else:
        hurricNamed_bundle = get_rdataset('hurricNamed', 'DAAG')
        with open(hurricNamed_pickle, 'wb') as hurricNamed_fp:
            pickle.dump(hurricNamed_bundle, hurricNamed_fp)
    hurricNamed_data = hurricNamed_bundle.data
    logger.info('hurricNamed data has variables %s' % list(hurricNamed_data))
    logger.info('hurricNamed data has %d rows and %d variables' % hurricNamed_data.shape)
    hurricNamed_title = hurricNamed_bundle.title
    logger.info('hurricNamed data has title %s' % hurricNamed_title)

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

    logger.info('loading Islay quartzite data')
    islay_pickle = data_folder + 'islay.pkl'
    if exists(islay_pickle):
        with open(islay_pickle, 'rb') as islay_fp:
            islay_bundle = pickle.load(islay_fp)
    else:
        islay_bundle = get_rdataset('islay', 'boot')
        with open(islay_pickle, 'wb') as islay_fp:
            pickle.dump(islay_bundle, islay_fp)
    islay_data = islay_bundle.data
    logger.info('islay data has variables %s and has %d rows' % (list(islay_data), len(islay_data)))
    islay_title = islay_bundle.title
    logger.info('islay data has title %s' % islay_title)

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

    logger.info('loading migraine headache treatment data')
    KosteckiDillon_pickle = data_folder + 'KosteckiDillon.pkl'
    if exists(KosteckiDillon_pickle):
        with open(KosteckiDillon_pickle, 'rb') as KosteckiDillon_fp:
            KosteckiDillon_bundle = pickle.load(KosteckiDillon_fp)
    else:
        KosteckiDillon_bundle = get_rdataset('KosteckiDillon', 'carData')
        with open(KosteckiDillon_pickle, 'wb') as KosteckiDillon_fp:
            pickle.dump(KosteckiDillon_bundle, KosteckiDillon_fp)
    KosteckiDillon_data = KosteckiDillon_bundle.data
    logger.info(
        'KosteckiDillon data has variables %s and has %d rows' % (list(KosteckiDillon_data), len(KosteckiDillon_data)))
    KosteckiDillon_title = KosteckiDillon_bundle.title
    logger.info('KosteckiDillon data has title %s' % KosteckiDillon_title)

    logger.info('loading lbw data')
    lbw_pickle = data_folder + 'lbw.pkl'
    if exists(lbw_pickle):
        with open(lbw_pickle, 'rb') as lbw_fp:
            lbw_bundle = pickle.load(lbw_fp)
    else:
        lbw_bundle = get_rdataset('lbw', 'COUNT')
        with open(lbw_pickle, 'wb') as lbw_fp:
            pickle.dump(lbw_bundle, lbw_fp)
    lbw_data = lbw_bundle.data
    logger.info('lbw data has variables %s and has %d rows' % (list(lbw_data), len(lbw_data)))
    lbw_title = lbw_bundle.title
    logger.info('lbw data has title %s' % lbw_title)

    logger.info('loading lbwgrp data')
    lbwgrp_pickle = data_folder + 'lbwgrp.pkl'
    if exists(lbwgrp_pickle):
        with open(lbwgrp_pickle, 'rb') as lbwgrp_fp:
            lbwgrp_bundle = pickle.load(lbwgrp_fp)
    else:
        lbwgrp_bundle = get_rdataset('lbwgrp', 'COUNT')
        with open(lbwgrp_pickle, 'wb') as lbwgrp_fp:
            pickle.dump(lbwgrp_bundle, lbwgrp_fp)
    lbwgrp_data = lbwgrp_bundle.data
    logger.info('lbwgrp data has variables %s and has %d rows' % (list(lbwgrp_data), len(lbwgrp_data)))
    lbwgrp_title = lbwgrp_bundle.title
    logger.info('lbwgrp data has title %s' % lbwgrp_title)

    logger.info('loading infant mortality data')
    Leinhardt_pickle = data_folder + 'Leinhardt.pkl'
    if exists(Leinhardt_pickle):
        with open(Leinhardt_pickle, 'rb') as Leinhardt_fp:
            Leinhardt_bundle = pickle.load(Leinhardt_fp)
    else:
        Leinhardt_bundle = get_rdataset('Leinhardt', 'carData')
        with open(Leinhardt_pickle, 'wb') as Leinhardt_fp:
            pickle.dump(Leinhardt_bundle, Leinhardt_fp)
    Leinhardt_data = Leinhardt_bundle.data
    logger.info(
        'Leinhardt data has variables %s and has %d rows' % (list(Leinhardt_data), len(Leinhardt_data)))
    Leinhardt_title = Leinhardt_bundle.title
    logger.info('Leinhardt data has title %s' % Leinhardt_title)

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

    logger.info('loading cancer drug (skew power distribution) data')
    LoBD_pickle = data_folder + 'LoBD.pkl'
    if exists(LoBD_pickle):
        with open(LoBD_pickle, 'rb') as LoBD_fp:
            LoBD_bundle = pickle.load(LoBD_fp)
    else:
        LoBD_bundle = get_rdataset('LoBD', 'carData')
        with open(LoBD_pickle, 'wb') as LoBD_fp:
            pickle.dump(LoBD_bundle, LoBD_fp)
    LoBD_data = LoBD_bundle.data
    logger.info(
        'LoBD data has variables %s and has %d rows' % (list(LoBD_data), len(LoBD_data)))
    LoBD_title = LoBD_bundle.title
    logger.info('LoBD data has title %s' % LoBD_title)

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

    logger.info('loading loomis data')
    loomis_pickle = data_folder + 'loomis.pkl'
    if exists(loomis_pickle):
        with open(loomis_pickle, 'rb') as loomis_fp:
            loomis_bundle = pickle.load(loomis_fp)
    else:
        loomis_bundle = get_rdataset('loomis', 'COUNT')
        with open(loomis_pickle, 'wb') as loomis_fp:
            pickle.dump(loomis_bundle, loomis_fp)
    loomis_data = loomis_bundle.data
    logger.info('loomis data has variables %s and has %d rows' % (list(loomis_data), len(loomis_data)))
    loomis_title = loomis_bundle.title
    logger.info('loomis data has title %s' % loomis_title)

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

    logger.info('loading Manaus river height data')
    manaus_pickle = data_folder + 'manaus.pkl'
    if exists(manaus_pickle):
        with open(manaus_pickle, 'rb') as manaus_fp:
            manaus_bundle = pickle.load(manaus_fp)
    else:
        manaus_bundle = get_rdataset('manaus', 'boot')
        with open(manaus_pickle, 'wb') as manaus_fp:
            pickle.dump(manaus_bundle, manaus_fp)
    manaus_data = manaus_bundle.data
    logger.info('manaus data has variables %s and has %d rows' % (list(manaus_data), len(manaus_data)))
    manaus_title = manaus_bundle.title
    logger.info('manaus data has title %s' % manaus_title)

    logger.info('loading contrived collinear data')
    Mandel_pickle = data_folder + 'Mandel.pkl'
    if exists(Mandel_pickle):
        with open(Mandel_pickle, 'rb') as Mandel_fp:
            Mandel_bundle = pickle.load(Mandel_fp)
    else:
        Mandel_bundle = get_rdataset('Mandel', 'carData')
        with open(Mandel_pickle, 'wb') as Mandel_fp:
            pickle.dump(Mandel_bundle, Mandel_fp)
    Mandel_data = Mandel_bundle.data
    logger.info(
        'Mandel data has variables %s and has %d rows' % (list(Mandel_data), len(Mandel_data)))
    Mandel_title = Mandel_bundle.title
    logger.info('Mandel data has title %s' % Mandel_title)

    logger.info('loading mdvis data')
    mdvis_pickle = data_folder + 'mdvis.pkl'
    if exists(mdvis_pickle):
        with open(mdvis_pickle, 'rb') as mdvis_fp:
            mdvis_bundle = pickle.load(mdvis_fp)
    else:
        mdvis_bundle = get_rdataset('mdvis', 'COUNT')
        with open(mdvis_pickle, 'wb') as mdvis_fp:
            pickle.dump(mdvis_bundle, mdvis_fp)
    mdvis_data = mdvis_bundle.data
    logger.info('mdvis data has variables %s and has %d rows' % (list(mdvis_data), len(mdvis_data)))
    mdvis_title = mdvis_bundle.title
    logger.info('mdvis data has title %s' % mdvis_title)

    logger.info('loading medpar data')
    medpar_pickle = data_folder + 'medpar.pkl'
    if exists(medpar_pickle):
        with open(medpar_pickle, 'rb') as medpar_fp:
            medpar_bundle = pickle.load(medpar_fp)
    else:
        medpar_bundle = get_rdataset('medpar', 'COUNT')
        with open(medpar_pickle, 'wb') as medpar_fp:
            pickle.dump(medpar_bundle, medpar_fp)
    medpar_data = medpar_bundle.data
    logger.info('medpar data has variables %s and has %d rows' % (list(medpar_data), len(medpar_data)))
    medpar_title = medpar_bundle.title
    logger.info('medpar data has title %s' % medpar_title)

    logger.info('loading melanoma survival data')
    melanoma_pickle = data_folder + 'melanoma.pkl'
    if exists(melanoma_pickle):
        with open(melanoma_pickle, 'rb') as melanoma_fp:
            melanoma_bundle = pickle.load(melanoma_fp)
    else:
        melanoma_bundle = get_rdataset('melanoma', 'boot')
        with open(melanoma_pickle, 'wb') as melanoma_fp:
            pickle.dump(melanoma_bundle, melanoma_fp)
    melanoma_data = melanoma_bundle.data
    logger.info('melanoma data has variables %s and has %d rows' % (list(melanoma_data), len(melanoma_data)))
    melanoma_title = melanoma_bundle.title
    logger.info('melanoma data has title %s' % melanoma_title)

    logger.info('loading Canadian migration data')
    Migration_pickle = data_folder + 'Migration.pkl'
    if exists(Migration_pickle):
        with open(Migration_pickle, 'rb') as Migration_fp:
            Migration_bundle = pickle.load(Migration_fp)
    else:
        Migration_bundle = get_rdataset('Migration', 'carData')
        with open(Migration_pickle, 'wb') as Migration_fp:
            pickle.dump(Migration_bundle, Migration_fp)
    Migration_data = Migration_bundle.data
    logger.info(
        'Migration data has variables %s and has %d rows' % (list(Migration_data), len(Migration_data)))
    Migration_title = Migration_bundle.title
    logger.info('Migration data has title %s' % Migration_title)

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

    logger.info('loading status, authoritarianism, and conformity data')
    Moore_pickle = data_folder + 'Moore.pkl'
    if exists(Moore_pickle):
        with open(Moore_pickle, 'rb') as Moore_fp:
            Moore_bundle = pickle.load(Moore_fp)
    else:
        Moore_bundle = get_rdataset('Moore', 'carData')
        with open(Moore_pickle, 'wb') as Moore_fp:
            pickle.dump(Moore_bundle, Moore_fp)
    Moore_data = Moore_bundle.data
    logger.info(
        'Moore data has variables %s and has %d rows' % (list(Moore_data), len(Moore_data)))
    Moore_title = Moore_bundle.title
    logger.info('Moore data has title %s' % Moore_title)

    logger.info('loading simulated motorcycle accident data')
    motor_pickle = data_folder + 'motor.pkl'
    if exists(motor_pickle):
        with open(motor_pickle, 'rb') as motor_fp:
            motor_bundle = pickle.load(motor_fp)
    else:
        motor_bundle = get_rdataset('motor', 'boot')
        with open(motor_pickle, 'wb') as motor_fp:
            pickle.dump(motor_bundle, motor_fp)
    motor_data = motor_bundle.data
    logger.info('motor data has variables %s and has %d rows' % (list(motor_data), len(motor_data)))
    motor_title = motor_bundle.title
    logger.info('motor data has title %s' % motor_title)

    logger.info('loading Minneapolis 2015 demographic by neighborhood data')
    MplsDemo_pickle = data_folder + 'MplsDemo.pkl'
    if exists(MplsDemo_pickle):
        with open(MplsDemo_pickle, 'rb') as MplsDemo_fp:
            MplsDemo_bundle = pickle.load(MplsDemo_fp)
    else:
        MplsDemo_bundle = get_rdataset('MplsDemo', 'carData')
        with open(MplsDemo_pickle, 'wb') as MplsDemo_fp:
            pickle.dump(MplsDemo_bundle, MplsDemo_fp)
    MplsDemo_data = MplsDemo_bundle.data
    logger.info(
        'MplsDemo data has variables %s and has %d rows' % (list(MplsDemo_data), len(MplsDemo_data)))
    MplsDemo_title = MplsDemo_bundle.title
    logger.info('MplsDemo data has title %s' % MplsDemo_title)

    logger.info('loading Minneapolis 2018 police stop data')
    MplsStops_pickle = data_folder + 'MplsStops.pkl'
    if exists(MplsStops_pickle):
        with open(MplsStops_pickle, 'rb') as MplsStops_fp:
            MplsStops_bundle = pickle.load(MplsStops_fp)
    else:
        MplsStops_bundle = get_rdataset('MplsStops', 'carData')
        with open(MplsStops_pickle, 'wb') as MplsStops_fp:
            pickle.dump(MplsStops_bundle, MplsStops_fp)
    MplsStops_data = MplsStops_bundle.data
    logger.info(
        'MplsStops data has variables %s and has %d rows' % (list(MplsStops_data), len(MplsStops_data)))
    MplsStops_title = MplsStops_bundle.title
    logger.info('MplsStops data has title %s' % MplsStops_title)

    logger.info('loading US Womens labor participation data')
    Mroz_pickle = data_folder + 'Mroz.pkl'
    if exists(Mroz_pickle):
        with open(Mroz_pickle, 'rb') as Mroz_fp:
            Mroz_bundle = pickle.load(Mroz_fp)
    else:
        Mroz_bundle = get_rdataset('Mroz', 'carData')
        with open(Mroz_pickle, 'wb') as Mroz_fp:
            pickle.dump(Mroz_bundle, Mroz_fp)
    Mroz_data = Mroz_bundle.data
    logger.info('Mroz data has variables %s and has %d rows' % (list(Mroz_data), len(Mroz_data)))
    Mroz_title = Mroz_bundle.title
    logger.info('Mroz data has title %s' % Mroz_title)

    logger.info('loading neurophysiological point process data')
    neuro_pickle = data_folder + 'neuro.pkl'
    if exists(neuro_pickle):
        with open(neuro_pickle, 'rb') as neuro_fp:
            neuro_bundle = pickle.load(neuro_fp)
    else:
        neuro_bundle = get_rdataset('neuro', 'boot')
        with open(neuro_pickle, 'wb') as neuro_fp:
            pickle.dump(neuro_bundle, neuro_fp)
    neuro_data = neuro_bundle.data
    logger.info('neuro data has variables %s and has %d rows' % (list(neuro_data), len(neuro_data)))
    neuro_title = neuro_bundle.title
    logger.info('neuro data has title %s' % neuro_title)

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

    logger.info('loading aquatic nitrofen data')
    nitrofen_pickle = data_folder + 'nitrofen.pkl'
    if exists(nitrofen_pickle):
        with open(nitrofen_pickle, 'rb') as nitrofen_fp:
            nitrofen_bundle = pickle.load(nitrofen_fp)
    else:
        nitrofen_bundle = get_rdataset('nitrofen', 'boot')
        with open(nitrofen_pickle, 'wb') as nitrofen_fp:
            pickle.dump(nitrofen_bundle, nitrofen_fp)
    nitrofen_data = nitrofen_bundle.data
    logger.info('nitrofen data has variables %s and has %d rows' % (list(nitrofen_data), len(nitrofen_data)))
    nitrofen_title = nitrofen_bundle.title
    logger.info('nitrofen data has title %s' % nitrofen_title)

    logger.info('loading prostate cancer data')
    nodal_pickle = data_folder + 'nodal.pkl'
    if exists(nodal_pickle):
        with open(nodal_pickle, 'rb') as nodal_fp:
            nodal_bundle = pickle.load(nodal_fp)
    else:
        nodal_bundle = get_rdataset('nodal', 'boot')
        with open(nodal_pickle, 'wb') as nodal_fp:
            pickle.dump(nodal_bundle, nodal_fp)
    nodal_data = nodal_bundle.data
    logger.info('nodal data has variables %s and has %d rows' % (list(nodal_data), len(nodal_data)))
    nodal_title = nodal_bundle.title
    logger.info('nodal data has title %s' % nodal_title)

    logger.info('loading nuclear power station construction data')
    nuclear_pickle = data_folder + 'nuclear.pkl'
    if exists(nuclear_pickle):
        with open(nuclear_pickle, 'rb') as nuclear_fp:
            nuclear_bundle = pickle.load(nuclear_fp)
    else:
        nuclear_bundle = get_rdataset('nuclear', 'boot')
        with open(nuclear_pickle, 'wb') as nuclear_fp:
            pickle.dump(nuclear_bundle, nuclear_fp)
    nuclear_data = nuclear_bundle.data
    logger.info('nuclear data has variables %s and has %d rows' % (list(nuclear_data), len(nuclear_data)))
    nuclear_title = nuclear_bundle.title
    logger.info('nuclear data has title %s' % nuclear_title)

    logger.info('loading nuts data')
    nuts_pickle = data_folder + 'nuts.pkl'
    if exists(nuts_pickle):
        with open(nuts_pickle, 'rb') as nuts_fp:
            nuts_bundle = pickle.load(nuts_fp)
    else:
        nuts_bundle = get_rdataset('nuts', 'COUNT')
        with open(nuts_pickle, 'wb') as nuts_fp:
            pickle.dump(nuts_bundle, nuts_fp)
    nuts_data = nuts_bundle.data
    logger.info('nuts data has variables %s and has %d rows' % (list(nuts_data), len(nuts_data)))
    nuts_title = nuts_bundle.title
    logger.info('nuts data has title %s' % nuts_title)

    logger.info('loading repeated measures data')
    OBrienKaiser_pickle = data_folder + 'OBrienKaiser.pkl'
    if exists(OBrienKaiser_pickle):
        with open(OBrienKaiser_pickle, 'rb') as OBrienKaiser_fp:
            OBrienKaiser_bundle = pickle.load(OBrienKaiser_fp)
    else:
        OBrienKaiser_bundle = get_rdataset('OBrienKaiser', 'carData')
        with open(OBrienKaiser_pickle, 'wb') as OBrienKaiser_fp:
            pickle.dump(OBrienKaiser_bundle, OBrienKaiser_fp)
    OBrienKaiser_data = OBrienKaiser_bundle.data
    logger.info(
        'OBrienKaiser data has variables %s and has %d rows' % (list(OBrienKaiser_data), len(OBrienKaiser_data)))
    OBrienKaiser_title = OBrienKaiser_bundle.title
    logger.info('OBrienKaiser data has title %s' % OBrienKaiser_title)

    logger.info('loading Olivetti faces data')
    olivetti_faces = fetch_olivetti_faces(data_home=data_folder)
    olivetti_faces_data = olivetti_faces['data']
    olivetti_faces_images = olivetti_faces['images']
    olivetti_faces_target = olivetti_faces['target']
    olivetti_faces_description = olivetti_faces['DESCR']

    logger.info('loading Canadian directorates data')
    Ornstein_pickle = data_folder + 'Ornstein.pkl'
    if exists(Ornstein_pickle):
        with open(Ornstein_pickle, 'rb') as Ornstein_fp:
            Ornstein_bundle = pickle.load(Ornstein_fp)
    else:
        Ornstein_bundle = get_rdataset('Ornstein', 'carData')
        with open(Ornstein_pickle, 'wb') as Ornstein_fp:
            pickle.dump(Ornstein_bundle, Ornstein_fp)
    Ornstein_data = Ornstein_bundle.data
    logger.info('Ornstein data has variables %s and has %d rows' % (list(Ornstein_data), len(Ornstein_data)))
    Ornstein_title = Ornstein_bundle.title
    logger.info('Ornstein data has title %s' % Ornstein_title)

    logger.info('loading Guinea pig brain data')
    paulsen_pickle = data_folder + 'paulsen.pkl'
    if exists(paulsen_pickle):
        with open(paulsen_pickle, 'rb') as paulsen_fp:
            paulsen_bundle = pickle.load(paulsen_fp)
    else:
        paulsen_bundle = get_rdataset('paulsen', 'boot')
        with open(paulsen_pickle, 'wb') as paulsen_fp:
            pickle.dump(paulsen_bundle, paulsen_fp)
    paulsen_data = paulsen_bundle.data
    logger.info('paulsen data has variables %s and has %d rows' % (list(paulsen_data), len(paulsen_data)))
    paulsen_title = paulsen_bundle.title
    logger.info('paulsen data has title %s' % paulsen_title)

    logger.info('loading plant species traits data')
    plantTraits_pickle = data_folder + 'plantTraits.pkl'
    if exists(plantTraits_pickle):
        with open(plantTraits_pickle, 'rb') as plantTraits_fp:
            plantTraits_bundle = pickle.load(plantTraits_fp)
    else:
        plantTraits_bundle = get_rdataset('plantTraits', 'cluster')
        with open(plantTraits_pickle, 'wb') as plantTraits_fp:
            pickle.dump(plantTraits_bundle, plantTraits_fp)
    plantTraits_data = plantTraits_bundle.data
    logger.info('plantTraits data has variables %s and has %d rows' % (list(plantTraits_data), len(plantTraits_data)))
    plantTraits_title = plantTraits_bundle.title
    logger.info('plantTraits data has title %s' % plantTraits_title)

    logger.info('loading plant species traits data')
    plantTraits_pickle = data_folder + 'plantTraits.pkl'
    if exists(plantTraits_pickle):
        with open(plantTraits_pickle, 'rb') as plantTraits_fp:
            plantTraits_bundle = pickle.load(plantTraits_fp)
    else:
        plantTraits_bundle = get_rdataset('plantTraits', 'cluster')
        with open(plantTraits_pickle, 'wb') as plantTraits_fp:
            pickle.dump(plantTraits_bundle, plantTraits_fp)
    plantTraits_data = plantTraits_bundle.data
    logger.info('plantTraits data has variables %s and has %d rows' % (list(plantTraits_data), len(plantTraits_data)))
    plantTraits_title = plantTraits_bundle.title
    logger.info('plantTraits data has title %s' % plantTraits_title)

    logger.info('loading plutonium batch data')
    pluton_pickle = data_folder + 'pluton.pkl'
    if exists(pluton_pickle):
        with open(pluton_pickle, 'rb') as pluton_fp:
            pluton_bundle = pickle.load(pluton_fp)
    else:
        pluton_bundle = get_rdataset('pluton', 'cluster')
        with open(pluton_pickle, 'wb') as pluton_fp:
            pickle.dump(pluton_bundle, pluton_fp)
    pluton_data = pluton_bundle.data
    logger.info('pluton data has variables %s and has %d rows' % (list(pluton_data), len(pluton_data)))
    pluton_title = pluton_bundle.title
    logger.info('pluton data has title %s' % pluton_title)

    logger.info('loading animal poison data')
    poisons_pickle = data_folder + 'poisons.pkl'
    if exists(poisons_pickle):
        with open(poisons_pickle, 'rb') as poisons_fp:
            poisons_bundle = pickle.load(poisons_fp)
    else:
        poisons_bundle = get_rdataset('poisons', 'boot')
        with open(poisons_pickle, 'wb') as poisons_fp:
            pickle.dump(poisons_bundle, poisons_fp)
    poisons_data = poisons_bundle.data
    logger.info('poisons data has variables %s and has %d rows' % (list(poisons_data), len(poisons_data)))
    poisons_title = poisons_bundle.title
    logger.info('poisons data has title %s' % poisons_title)

    logger.info('loading New Caledonian Laterites data')
    polar_pickle = data_folder + 'polar.pkl'
    if exists(polar_pickle):
        with open(polar_pickle, 'rb') as polar_fp:
            polar_bundle = pickle.load(polar_fp)
    else:
        polar_bundle = get_rdataset('polar', 'boot')
        with open(polar_pickle, 'wb') as polar_fp:
            pickle.dump(polar_bundle, polar_fp)
    polar_data = polar_bundle.data
    logger.info('polar data has variables %s and has %d rows' % (list(polar_data), len(polar_data)))
    polar_title = polar_bundle.title
    logger.info('polar data has title %s' % polar_title)

    logger.info('loading chemical composition of pottery data')
    Pottery_pickle = data_folder + 'Pottery.pkl'
    if exists(Pottery_pickle):
        with open(Pottery_pickle, 'rb') as Pottery_fp:
            Pottery_bundle = pickle.load(Pottery_fp)
    else:
        Pottery_bundle = get_rdataset('Pottery', 'carData')
        with open(Pottery_pickle, 'wb') as Pottery_fp:
            pickle.dump(Pottery_bundle, Pottery_fp)
    Pottery_data = Pottery_bundle.data
    logger.info('Pottery data has variables %s and has %d rows' % (list(Pottery_data), len(Pottery_data)))
    Pottery_title = Pottery_bundle.title
    logger.info('Pottery data has title %s' % Pottery_title)

    logger.info('loading Canadian occupation prestige data')
    Prestige_pickle = data_folder + 'Prestige.pkl'
    if exists(Prestige_pickle):
        with open(Prestige_pickle, 'rb') as Prestige_fp:
            Prestige_bundle = pickle.load(Prestige_fp)
    else:
        Prestige_bundle = get_rdataset('Prestige', 'carData')
        with open(Prestige_pickle, 'wb') as Prestige_fp:
            pickle.dump(Prestige_bundle, Prestige_fp)
    Prestige_data = Prestige_bundle.data
    logger.info('Prestige data has variables %s and has %d rows' % (list(Prestige_data), len(Prestige_data)))
    Prestige_title = Prestige_bundle.title
    logger.info('Prestige data has title %s' % Prestige_title)

    logger.info('loading demo regression data')
    Quartet_pickle = data_folder + 'Quartet.pkl'
    if exists(Quartet_pickle):
        with open(Quartet_pickle, 'rb') as Quartet_fp:
            Quartet_bundle = pickle.load(Quartet_fp)
    else:
        Quartet_bundle = get_rdataset('Quartet', 'carData')
        with open(Quartet_pickle, 'wb') as Quartet_fp:
            pickle.dump(Quartet_bundle, Quartet_fp)
    Quartet_data = Quartet_bundle.data
    logger.info('Quartet data has variables %s and has %d rows' % (list(Quartet_data), len(Quartet_data)))
    Quartet_title = Quartet_bundle.title
    logger.info('Quartet data has title %s' % Quartet_title)

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

    logger.info('loading cancer remission data')
    remission_pickle = data_folder + 'remission.pkl'
    if exists(remission_pickle):
        with open(remission_pickle, 'rb') as remission_fp:
            remission_bundle = pickle.load(remission_fp)
    else:
        remission_bundle = get_rdataset('remission', 'boot')
        with open(remission_pickle, 'wb') as remission_fp:
            pickle.dump(remission_bundle, remission_fp)
    remission_data = remission_bundle.data
    logger.info('remission data has variables %s and has %d rows' % (list(remission_data), len(remission_data)))
    remission_title = remission_bundle.title
    logger.info('remission data has title %s' % remission_title)

    logger.info('loading Reuters Corpus Volume I data')
    reuters_pickle = data_folder + 'reuters.pkl'
    rcv1_bunch = fetch_rcv1(subset='all', download_if_missing=True, random_state=random_state)
    rcv1_data = rcv1_bunch['data']
    rcv1_target = rcv1_bunch['target']
    rcv1_sample_id = rcv1_bunch['sample_id']
    rcv1_target_names = rcv1_bunch['target_names']
    rcv1_description = rcv1_bunch['DESCR']
    logger.info('Reuters data has description %s' % str(rcv1_description).strip())

    logger.info('loading fertility and contraception data')
    Robey_pickle = data_folder + 'Robey.pkl'
    if exists(Robey_pickle):
        with open(Robey_pickle, 'rb') as Robey_fp:
            Robey_bundle = pickle.load(Robey_fp)
    else:
        Robey_bundle = get_rdataset('Robey', 'carData')
        with open(Robey_pickle, 'wb') as Robey_fp:
            pickle.dump(Robey_bundle, Robey_fp)
    Robey_data = Robey_bundle.data
    logger.info('Robey data has variables %s and has %d rows' % (list(Robey_data), len(Robey_data)))
    Robey_title = Robey_bundle.title
    logger.info('Robey data has title %s' % Robey_title)

    logger.info('loading Ruspini data')
    ruspini_pickle = data_folder + 'ruspini.pkl'
    if exists(ruspini_pickle):
        with open(ruspini_pickle, 'rb') as ruspini_fp:
            ruspini_bundle = pickle.load(ruspini_fp)
    else:
        ruspini_bundle = get_rdataset('ruspini', 'cluster')
        with open(ruspini_pickle, 'wb') as ruspini_fp:
            pickle.dump(ruspini_bundle, ruspini_fp)
    ruspini_data = ruspini_bundle.data
    logger.info('ruspini data has variables %s and has %d rows' % (list(ruspini_data), len(ruspini_data)))
    ruspini_title = ruspini_bundle.title
    logger.info('ruspini data has title %s' % ruspini_title)

    logger.info('loading rwm data')
    rwm_pickle = data_folder + 'rwm.pkl'
    if exists(rwm_pickle):
        with open(rwm_pickle, 'rb') as rwm_fp:
            rwm_bundle = pickle.load(rwm_fp)
    else:
        rwm_bundle = get_rdataset('rwm', 'COUNT')
        with open(rwm_pickle, 'wb') as rwm_fp:
            pickle.dump(rwm_bundle, rwm_fp)
    rwm_data = rwm_bundle.data
    logger.info('rwm data has variables %s and has %d rows' % (list(rwm_data), len(rwm_data)))
    rwm_title = rwm_bundle.title
    logger.info('rwm data has title %s' % rwm_title)

    logger.info('loading rwm1984 data')
    rwm1984_pickle = data_folder + 'rwm1984.pkl'
    if exists(rwm1984_pickle):
        with open(rwm1984_pickle, 'rb') as rwm1984_fp:
            rwm1984_bundle = pickle.load(rwm1984_fp)
    else:
        rwm1984_bundle = get_rdataset('rwm1984', 'COUNT')
        with open(rwm1984_pickle, 'wb') as rwm1984_fp:
            pickle.dump(rwm1984_bundle, rwm1984_fp)
    rwm1984_data = rwm1984_bundle.data
    logger.info('rwm1984 data has variables %s and has %d rows' % (list(rwm1984_data), len(rwm1984_data)))
    rwm1984_title = rwm1984_bundle.title
    logger.info('rwm1984 data has title %s' % rwm1984_title)

    logger.info('loading rwm5yr data')
    rwm5yr_pickle = data_folder + 'rwm5yr.pkl'
    if exists(rwm5yr_pickle):
        with open(rwm5yr_pickle, 'rb') as rwm5yr_fp:
            rwm5yr_bundle = pickle.load(rwm5yr_fp)
    else:
        rwm5yr_bundle = get_rdataset('rwm5yr', 'COUNT')
        with open(rwm5yr_pickle, 'wb') as rwm5yr_fp:
            pickle.dump(rwm5yr_bundle, rwm5yr_fp)
    rwm5yr_data = rwm5yr_bundle.data
    logger.info('rwm5yr data has variables %s and has %d rows' % (list(rwm5yr_data), len(rwm5yr_data)))
    rwm5yr_title = rwm5yr_bundle.title
    logger.info('rwm5yr data has title %s' % rwm5yr_title)

    logger.info('loading Mazulu agriculture data')
    Sahlins_pickle = data_folder + 'Sahlins.pkl'
    if exists(Sahlins_pickle):
        with open(Sahlins_pickle, 'rb') as Sahlins_fp:
            Sahlins_bundle = pickle.load(Sahlins_fp)
    else:
        Sahlins_bundle = get_rdataset('Sahlins', 'carData')
        with open(Sahlins_pickle, 'wb') as Sahlins_fp:
            pickle.dump(Sahlins_bundle, Sahlins_fp)
    Sahlins_data = Sahlins_bundle.data
    logger.info('Sahlins data has variables %s and has %d rows' % (list(Sahlins_data), len(Sahlins_data)))
    Sahlins_title = Sahlins_bundle.title
    logger.info('Sahlins data has title %s' % Sahlins_title)

    logger.info('loading professor salary data')
    Salaries_pickle = data_folder + 'Salaries.pkl'
    if exists(Salaries_pickle):
        with open(Salaries_pickle, 'rb') as Salaries_fp:
            Salaries_bundle = pickle.load(Salaries_fp)
    else:
        Salaries_bundle = get_rdataset('Salaries', 'carData')
        with open(Salaries_pickle, 'wb') as Salaries_fp:
            pickle.dump(Salaries_bundle, Salaries_fp)
    Salaries_data = Salaries_bundle.data
    logger.info('Salaries data has variables %s and has %d rows' % (list(Salaries_data), len(Salaries_data)))
    Salaries_title = Salaries_bundle.title
    logger.info('Salaries data has title %s' % Salaries_title)

    logger.info('loading water salinity data')
    salinity_pickle = data_folder + 'salinity.pkl'
    if exists(salinity_pickle):
        with open(salinity_pickle, 'rb') as salinity_fp:
            salinity_bundle = pickle.load(salinity_fp)
    else:
        salinity_bundle = get_rdataset('salinity', 'boot')
        with open(salinity_pickle, 'wb') as salinity_fp:
            pickle.dump(salinity_bundle, salinity_fp)
    salinity_data = salinity_bundle.data
    logger.info('salinity data has variables %s and has %d rows' % (list(salinity_data), len(salinity_data)))
    salinity_title = salinity_bundle.title
    logger.info('salinity data has title %s' % salinity_title)

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
