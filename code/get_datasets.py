# http://scikit-learn.org/stable/datasets/index.html
# https://www.statsmodels.org/dev/datasets/index.html
import logging
import pickle
from os.path import exists
from time import time
from warnings import catch_warnings
from warnings import filterwarnings

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
    return_X_y = False

    logger.info('loading acme data')
    acme_pickle = data_folder + 'acme.pkl'
    if exists(acme_pickle):
        with open(acme_pickle, 'rb') as acme_fp:
            acme_bundle = pickle.load(acme_fp)
    else:
        acme_bundle = get_rdataset('acme', 'boot')
        with open(acme_pickle, 'wb') as acme_fp:
            pickle.dump(acme_bundle, acme_fp)
    acme_data = acme_bundle.data
    logger.info('acme data has variables %s' % list(acme_data))
    logger.info('acme data has %d rows and %d variables' % acme_data.shape)
    acme_title = acme_bundle.title
    logger.info('acme data has title %s' % acme_title)

    logger.info('loading aids data')
    aids_pickle = data_folder + 'aids.pkl'
    if exists(aids_pickle):
        with open(aids_pickle, 'rb') as aids_fp:
            aids_bundle = pickle.load(aids_fp)
    else:
        aids_bundle = get_rdataset('aids', 'boot')
        with open(aids_pickle, 'wb') as aids_fp:
            pickle.dump(aids_bundle, aids_fp)
    aids_data = aids_bundle.data
    logger.info('aids data has variables %s' % list(aids_data))
    logger.info('aids data has %d rows and %d variables' % aids_data.shape)
    aids_title = aids_bundle.title
    logger.info('aids data has title %s' % aids_title)

    logger.info('loading air conditioning failure data')
    aircondit_pickle = data_folder + 'aircondit.pkl'
    if exists(aircondit_pickle):
        with open(aircondit_pickle, 'rb') as aircondit_fp:
            aircondit_bundle = pickle.load(aircondit_fp)
    else:
        aircondit_bundle = get_rdataset('aircondit', 'boot')
        with open(aircondit_pickle, 'wb') as aircondit_fp:
            pickle.dump(aircondit_bundle, aircondit_fp)
    aircondit_data = aircondit_bundle.data
    logger.info('aircondit data has variables %s' % list(aircondit_data))
    logger.info('aircondit data has %d rows and %d variables' % aircondit_data.shape)
    aircondit_title = aircondit_bundle.title
    logger.info('aircondit data has title %s' % aircondit_title)

    logger.info('loading air conditioning 7 failure data')
    aircondit7_pickle = data_folder + 'aircondit7.pkl'
    if exists(aircondit7_pickle):
        with open(aircondit7_pickle, 'rb') as aircondit7_fp:
            aircondit7_bundle = pickle.load(aircondit7_fp)
    else:
        aircondit7_bundle = get_rdataset('aircondit7', 'boot')
        with open(aircondit7_pickle, 'wb') as aircondit7_fp:
            pickle.dump(aircondit7_bundle, aircondit7_fp)
    aircondit7_data = aircondit7_bundle.data
    logger.info('aircondit7 data has variables %s' % list(aircondit7_data))
    logger.info('aircondit7 data has %d rows and %d variables' % aircondit7_data.shape)
    aircondit7_title = aircondit7_bundle.title
    logger.info('aircondit7 data has title %s' % aircondit7_title)

    logger.info('loading car speeding and warning sign data')
    amis_pickle = data_folder + 'amis.pkl'
    if exists(amis_pickle):
        with open(amis_pickle, 'rb') as amis_fp:
            amis_bundle = pickle.load(amis_fp)
    else:
        amis_bundle = get_rdataset('amis', 'boot')
        with open(amis_pickle, 'wb') as amis_fp:
            pickle.dump(amis_bundle, amis_fp)
    amis_data = amis_bundle.data
    logger.info('amis data has variables %s' % list(amis_data))
    amis_title = amis_bundle.title
    logger.info('amis data has title %s' % amis_title)

    logger.info('loading remission time for Acute Myelogenous Leukemia data')
    aml_pickle = data_folder + 'aml.pkl'
    if exists(aml_pickle):
        with open(aml_pickle, 'rb') as aml_fp:
            aml_bundle = pickle.load(aml_fp)
    else:
        aml_bundle = get_rdataset('aml', 'boot')
        with open(aml_pickle, 'wb') as aml_fp:
            pickle.dump(aml_bundle, aml_fp)
    aml_data = aml_bundle.data
    logger.info('aml data has variables %s' % list(aml_data))
    aml_title = aml_bundle.title
    logger.info('aml data has title %s' % aml_title)

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

    logger.info('loading beaver body temperature data')
    beaver_pickle = data_folder + 'beaver.pkl'
    if exists(beaver_pickle):
        with open(beaver_pickle, 'rb') as beaver_fp:
            beaver_bundle = pickle.load(beaver_fp)
    else:
        beaver_bundle = get_rdataset('beaver', 'boot')
        with open(beaver_pickle, 'wb') as beaver_fp:
            pickle.dump(beaver_bundle, beaver_fp)
    beaver_data = beaver_bundle.data
    logger.info('beaver data has variables %s' % list(beaver_data))
    beaver_title = beaver_bundle.title
    logger.info('beaver data has title %s' % beaver_title)

    logger.info('loading big city population data')
    bigcity_pickle = data_folder + 'bigcity.pkl'
    if exists(bigcity_pickle):
        with open(bigcity_pickle, 'rb') as bigcity_fp:
            bigcity_bundle = pickle.load(bigcity_fp)
    else:
        bigcity_bundle = get_rdataset('bigcity', 'boot')
        with open(bigcity_pickle, 'wb') as bigcity_fp:
            pickle.dump(bigcity_bundle, bigcity_fp)
    bigcity_data = bigcity_bundle.data
    logger.info('bigcity data has variables %s' % list(bigcity_data))
    bigcity_title = bigcity_bundle.title
    logger.info('bigcity data has title %s' % bigcity_title)

    logger.info('loading boston data')
    boston_bunch = load_boston(return_X_y=return_X_y)
    boston_data = boston_bunch.data
    logger.info('boston data is %d x %d' % boston_data.shape)
    boston_target = boston_bunch.target
    boston_feature_names = boston_bunch.feature_names
    logger.info('boston feature names: %s' % boston_feature_names)
    boston_description = boston_bunch.DESCR
    logger.debug('boston description: %s' % boston_description)

    logger.info('loading spatial location of bramble cane data')
    brambles_pickle = data_folder + 'brambles.pkl'
    if exists(brambles_pickle):
        with open(brambles_pickle, 'rb') as brambles_fp:
            brambles_bundle = pickle.load(brambles_fp)
    else:
        brambles_bundle = get_rdataset('brambles', 'boot')
        with open(brambles_pickle, 'wb') as brambles_fp:
            pickle.dump(brambles_bundle, brambles_fp)
    brambles_data = brambles_bundle.data
    logger.info('brambles data has variables %s' % list(brambles_data))
    brambles_title = brambles_bundle.title
    logger.info('brambles data has title %s' % brambles_title)

    logger.info('loading breast cancer data')
    breast_cancer_bunch = load_breast_cancer(return_X_y=return_X_y)
    breast_cancer_data = breast_cancer_bunch['data']
    logger.info('cancer data is %d x %d' % breast_cancer_data.shape)
    breast_cancer_target = breast_cancer_bunch['target']
    breast_cancer_feature_names = breast_cancer_bunch['feature_names']
    logger.info('cancer feature names are %s' % breast_cancer_feature_names)
    breast_cancer_description = breast_cancer_bunch['DESCR']
    logger.debug('cancer description: %s' % breast_cancer_description)

    logger.info('loading smoking deaths among doctors data')
    breslow_pickle = data_folder + 'breslow.pkl'
    if exists(breslow_pickle):
        with open(breslow_pickle, 'rb') as breslow_fp:
            breslow_bundle = pickle.load(breslow_fp)
    else:
        breslow_bundle = get_rdataset('breslow', 'boot')
        with open(breslow_pickle, 'wb') as breslow_fp:
            pickle.dump(breslow_bundle, breslow_fp)
    breslow_data = breslow_bundle.data
    logger.info('breslow data has variables %s' % list(breslow_data))
    breslow_title = breslow_bundle.title
    logger.info('breslow data has title %s' % breslow_title)

    logger.info('loading calcium uptake data')
    calcium_pickle = data_folder + 'calcium.pkl'
    if exists(calcium_pickle):
        with open(calcium_pickle, 'rb') as calcium_fp:
            calcium_bundle = pickle.load(calcium_fp)
    else:
        calcium_bundle = get_rdataset('calcium', 'boot')
        with open(calcium_pickle, 'wb') as calcium_fp:
            pickle.dump(calcium_bundle, calcium_fp)
    calcium_data = calcium_bundle.data
    logger.info('calcium data has variables %s' % list(calcium_data))
    calcium_title = calcium_bundle.title
    logger.info('calcium data has title %s' % calcium_title)

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

    logger.info('loading sugar-cane disease data')
    cane_pickle = data_folder + 'cane.pkl'
    if exists(cane_pickle):
        with open(cane_pickle, 'rb') as cane_fp:
            cane_bundle = pickle.load(cane_fp)
    else:
        cane_bundle = get_rdataset('cane', 'boot')
        with open(cane_pickle, 'wb') as cane_fp:
            pickle.dump(cane_bundle, cane_fp)
    cane_data = cane_bundle.data
    logger.info('cane data has variables %s' % list(cane_data))
    cane_title = cane_bundle.title
    logger.info('cane data has title %s' % cane_title)

    logger.info('loading simulated manufacturing process data')
    capability_pickle = data_folder + 'capability.pkl'
    if exists(capability_pickle):
        with open(capability_pickle, 'rb') as capability_fp:
            capability_bundle = pickle.load(capability_fp)
    else:
        capability_bundle = get_rdataset('capability', 'boot')
        with open(capability_pickle, 'wb') as capability_fp:
            pickle.dump(capability_bundle, capability_fp)
    capability_data = capability_bundle.data
    logger.info('capability data has variables %s' % list(capability_data))
    capability_title = capability_bundle.title
    logger.info('capability data has title %s' % capability_title)

    logger.info('loading credit card data')
    ccard_pickle = data_folder + 'ccard.pkl'
    if exists(ccard_pickle):
        with open(ccard_pickle, 'rb') as ccard_fp:
            ccard_bunch = pickle.load(ccard_fp)
    else:
        ccard_bunch = ccard.load_pandas()
        with open(ccard_pickle, 'wb') as ccard_fp:
            pickle.dump(ccard_bunch, ccard_fp)
    ccard_data = ccard_bunch['data']
    logger.info('Credit card data is %d x %d' % ccard_data.shape)
    ccard_names = ccard_bunch['names']
    ccard_endog = ccard_bunch['endog_name']
    logger.info('Credit card endogenous variable is %s' % ccard_endog)
    ccard_exog = ccard_bunch['exog_name']
    logger.info('Credit card exogenous variable is %s' % ccard_exog)

    logger.info('loading China smoking data')
    china_smoking_pickle = data_folder + 'china_smoking.pkl'
    if exists(china_smoking_pickle):
        with open(china_smoking_pickle, 'rb') as china_smoking_fp:
            china_smoking_bunch = pickle.load(china_smoking_fp)
    else:
        china_smoking_bunch = china_smoking.load_pandas()
        with open(china_smoking_pickle, 'wb') as china_smoking_fp:
            pickle.dump(china_smoking_bunch, china_smoking_fp)
    china_smoking_data = china_smoking_bunch['data']
    logger.info('China smoking data is %d x %d' % china_smoking_data.shape)
    china_smoking_title = china_smoking_bunch['title']

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
    logger.info('digits data is %d x %d' % diabetes_data.shape)
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

    logger.info('loading Heart transplant data')
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
