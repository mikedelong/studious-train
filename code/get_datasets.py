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

    logger.info('loading experimenter expectations data')
    Adler_pickle = data_folder + 'Adler.pkl'
    if exists(Adler_pickle):
        with open(Adler_pickle, 'rb') as Adler_fp:
            Adler_bundle = pickle.load(Adler_fp)
    else:
        Adler_bundle = get_rdataset('Adler', 'carData')
        with open(Adler_pickle, 'wb') as Adler_fp:
            pickle.dump(Adler_bundle, Adler_fp)
    Adler_data = Adler_bundle.data
    logger.info('Adler data has variables %s and has %d rows' % (list(Adler_data), len(Adler_data)))
    Adler_title = Adler_bundle.title
    logger.info('Adler data has title %s' % Adler_title)

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

    logger.info('loading AMS survey data')
    AMSsurvey_pickle = data_folder + 'AMSsurvey.pkl'
    if exists(AMSsurvey_pickle):
        with open(AMSsurvey_pickle, 'rb') as AMSsurvey_fp:
            AMSsurvey_bundle = pickle.load(AMSsurvey_fp)
    else:
        AMSsurvey_bundle = get_rdataset('AMSsurvey', 'carData')
        with open(AMSsurvey_pickle, 'wb') as AMSsurvey_fp:
            pickle.dump(AMSsurvey_bundle, AMSsurvey_fp)
    AMSsurvey_data = AMSsurvey_bundle.data
    logger.info('AMSsurvey data has variables %s and has %d rows' % (list(AMSsurvey_data), len(AMSsurvey_data)))
    AMSsurvey_title = AMSsurvey_bundle.title
    logger.info('AMSsurvey data has title %s' % AMSsurvey_title)

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

    logger.info('loading city integration data')
    Angell_pickle = data_folder + 'Angell.pkl'
    if exists(Angell_pickle):
        with open(Angell_pickle, 'rb') as Angell_fp:
            Angell_bundle = pickle.load(Angell_fp)
    else:
        Angell_bundle = get_rdataset('Angell', 'carData')
        with open(Angell_pickle, 'wb') as Angell_fp:
            pickle.dump(Angell_bundle, Angell_fp)
    Angell_data = Angell_bundle.data
    logger.info('Angell data has variables %s and has %d rows' % (list(Angell_data), len(Angell_data)))
    Angell_title = Angell_bundle.title
    logger.info('Angell data has title %s' % Angell_title)

    logger.info('loading US public school expenditure data')
    Anscombe_pickle = data_folder + 'Anscombe.pkl'
    if exists(Anscombe_pickle):
        with open(Anscombe_pickle, 'rb') as Anscombe_fp:
            Anscombe_bundle = pickle.load(Anscombe_fp)
    else:
        Anscombe_bundle = get_rdataset('Anscombe', 'carData')
        with open(Anscombe_pickle, 'wb') as Anscombe_fp:
            pickle.dump(Anscombe_bundle, Anscombe_fp)
    Anscombe_data = Anscombe_bundle.data
    logger.info('Anscombe data has variables %s and has %d rows' % (list(Anscombe_data), len(Anscombe_data)))
    Anscombe_title = Anscombe_bundle.title
    logger.info('Anscombe data has title %s' % Anscombe_title)

    logger.info('loading marijuana arrest data')
    Arrests_pickle = data_folder + 'Arrests.pkl'
    if exists(Arrests_pickle):
        with open(Arrests_pickle, 'rb') as Arrests_fp:
            Arrests_bundle = pickle.load(Arrests_fp)
    else:
        Arrests_bundle = get_rdataset('Arrests', 'carData')
        with open(Arrests_pickle, 'wb') as Arrests_fp:
            pickle.dump(Arrests_bundle, Arrests_fp)
    Arrests_data = Arrests_bundle.data
    logger.info('Arrests data has variables %s and has %d rows' % (list(Arrests_data), len(Arrests_data)))
    Arrests_title = Arrests_bundle.title
    logger.info('Arrests data has title %s' % Arrests_title)

    logger.info('loading reading comprehension data')
    Baumann_pickle = data_folder + 'Baumann.pkl'
    if exists(Baumann_pickle):
        with open(Baumann_pickle, 'rb') as Baumann_fp:
            Baumann_bundle = pickle.load(Baumann_fp)
    else:
        Baumann_bundle = get_rdataset('Baumann', 'carData')
        with open(Baumann_pickle, 'wb') as Baumann_fp:
            pickle.dump(Baumann_bundle, Baumann_fp)
    Baumann_data = Baumann_bundle.data
    logger.info('Baumann data has variables %s and has %d rows' % (list(Baumann_data), len(Baumann_data)))
    Baumann_title = Baumann_bundle.title
    logger.info('Baumann data has title %s' % Baumann_title)

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

    logger.info('loading British elections data')
    BEPS_pickle = data_folder + 'BEPS.pkl'
    if exists(BEPS_pickle):
        with open(BEPS_pickle, 'rb') as BEPS_fp:
            BEPS_bundle = pickle.load(BEPS_fp)
    else:
        BEPS_bundle = get_rdataset('BEPS', 'carData')
        with open(BEPS_pickle, 'wb') as BEPS_fp:
            pickle.dump(BEPS_bundle, BEPS_fp)
    BEPS_data = BEPS_bundle.data
    logger.info('BEPS data has variables %s and has %d rows' % (list(BEPS_data), len(BEPS_data)))
    BEPS_title = BEPS_bundle.title
    logger.info('BEPS data has title %s' % BEPS_title)

    logger.info('loading Canadian labor-force participation data')
    Bfox_pickle = data_folder + 'Bfox.pkl'
    if exists(Bfox_pickle):
        with open(Bfox_pickle, 'rb') as Bfox_fp:
            Bfox_bundle = pickle.load(Bfox_fp)
    else:
        Bfox_bundle = get_rdataset('Bfox', 'carData')
        with open(Bfox_pickle, 'wb') as Bfox_fp:
            pickle.dump(Bfox_bundle, Bfox_fp)
    Bfox_data = Bfox_bundle.data
    logger.info('Bfox data has variables %s and has %d rows' % (list(Bfox_data), len(Bfox_data)))
    Bfox_title = Bfox_bundle.title
    logger.info('Bfox data has title %s' % Bfox_title)

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

    logger.info('loading eating disorder exercise data')
    Blackmore_pickle = data_folder + 'Blackmore.pkl'
    if exists(Blackmore_pickle):
        with open(Blackmore_pickle, 'rb') as Blackmore_fp:
            Blackmore_bundle = pickle.load(Blackmore_fp)
    else:
        Blackmore_bundle = get_rdataset('Blackmore', 'carData')
        with open(Blackmore_pickle, 'wb') as Blackmore_fp:
            pickle.dump(Blackmore_bundle, Blackmore_fp)
    Blackmore_data = Blackmore_bundle.data
    logger.info('Blackmore data has variables %s and has %d rows' % (list(Blackmore_data), len(Blackmore_data)))
    Blackmore_title = Blackmore_bundle.title
    logger.info('Blackmore data has title %s' % Blackmore_title)

    logger.info('loading boston data')
    boston_bunch = load_boston(return_X_y=return_X_y)
    boston_data = boston_bunch.data
    logger.info('boston data is %d x %d' % boston_data.shape)
    boston_target = boston_bunch.target
    boston_feature_names = boston_bunch.feature_names
    logger.info('boston feature names: %s' % boston_feature_names)
    boston_description = boston_bunch.DESCR
    logger.debug('boston description: %s' % boston_description)

    logger.info('loading fake twins data')
    Burt_pickle = data_folder + 'Burt.pkl'
    if exists(Burt_pickle):
        with open(Burt_pickle, 'rb') as Burt_fp:
            Burt_bundle = pickle.load(Burt_fp)
    else:
        Burt_bundle = get_rdataset('Burt', 'carData')
        with open(Burt_pickle, 'wb') as Burt_fp:
            pickle.dump(Burt_bundle, Burt_fp)
    Burt_data = Burt_bundle.data
    logger.info('Burt data has variables %s and has %d rows' % (list(Burt_data), len(Burt_data)))
    Burt_title = Burt_bundle.title
    logger.info('Burt data has title %s' % Burt_title)

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

    logger.info('loading domestic cat weight data')
    catsM_pickle = data_folder + 'catsM.pkl'
    if exists(catsM_pickle):
        with open(catsM_pickle, 'rb') as catsM_fp:
            catsM_bundle = pickle.load(catsM_fp)
    else:
        catsM_bundle = get_rdataset('catsM', 'boot')
        with open(catsM_pickle, 'wb') as catsM_fp:
            pickle.dump(catsM_bundle, catsM_fp)
    catsM_data = catsM_bundle.data
    logger.info('catsM data has variables %s' % list(catsM_data))
    catsM_title = catsM_bundle.title
    logger.info('catsM data has title %s' % catsM_title)

    logger.info('loading muscle Caveolae position data')
    cav_pickle = data_folder + 'cav.pkl'
    if exists(cav_pickle):
        with open(cav_pickle, 'rb') as cav_fp:
            cav_bundle = pickle.load(cav_fp)
    else:
        cav_bundle = get_rdataset('cav', 'boot')
        with open(cav_pickle, 'wb') as cav_fp:
            pickle.dump(cav_bundle, cav_fp)
    cav_data = cav_bundle.data
    logger.info('cav data has variables %s' % list(cav_data))
    cav_title = cav_bundle.title
    logger.info('cav data has title %s' % cav_title)

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

    logger.info('loading AIDS patient CD4 count data')
    cd4_pickle = data_folder + 'cd4.pkl'
    if exists(cd4_pickle):
        with open(cd4_pickle, 'rb') as cd4_fp:
            cd4_bundle = pickle.load(cd4_fp)
    else:
        cd4_bundle = get_rdataset('cd4', 'boot')
        with open(cd4_pickle, 'wb') as cd4_fp:
            pickle.dump(cd4_bundle, cd4_fp)
    cd4_data = cd4_bundle.data
    logger.info('cd4 data has variables %s' % list(cd4_data))
    cd4_title = cd4_bundle.title
    logger.info('cd4 data has title %s' % cd4_title)

    logger.info('loading nested bootstrap of CD4 data')
    cd4_nested_pickle = data_folder + 'cd4.nested.pkl'
    if exists(cd4_nested_pickle):
        with open(cd4_nested_pickle, 'rb') as cd4_nested_fp:
            cd4_nested_bundle = pickle.load(cd4_nested_fp)
    else:
        cd4_nested_bundle = get_rdataset('cd4.nested', 'boot')
        with open(cd4_nested_pickle, 'wb') as cd4_nested_fp:
            pickle.dump(cd4_nested_bundle, cd4_nested_fp)
    cd4_nested_data = cd4_nested_bundle.data
    logger.info('cd4.nested data has variables %s' % list(cd4_nested_data))
    cd4_nested_title = cd4_nested_bundle.title
    logger.info('cd4.nested data has title %s' % cd4_nested_title)

    logger.info('loading Channing House data')
    channing_pickle = data_folder + 'channing.pkl'
    if exists(channing_pickle):
        with open(channing_pickle, 'rb') as channing_fp:
            channing_bundle = pickle.load(channing_fp)
    else:
        channing_bundle = get_rdataset('channing', 'boot')
        with open(channing_pickle, 'wb') as channing_fp:
            pickle.dump(channing_bundle, channing_fp)
    channing_data = channing_bundle.data
    logger.info('channing data has variables %s' % list(channing_data))
    channing_title = channing_bundle.title
    logger.info('channing data has title %s' % channing_title)

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

    logger.info('loading US City population data')
    city_pickle = data_folder + 'city.pkl'
    if exists(city_pickle):
        with open(city_pickle, 'rb') as city_fp:
            city_bundle = pickle.load(city_fp)
    else:
        city_bundle = get_rdataset('city', 'boot')
        with open(city_pickle, 'wb') as city_fp:
            pickle.dump(city_bundle, city_fp)
    city_data = city_bundle.data
    logger.info('city data has variables %s' % list(city_data))
    city_title = city_bundle.title
    logger.info('city data has title %s' % city_title)

    logger.info('loading Claridge left-handedness data')
    claridge_pickle = data_folder + 'claridge.pkl'
    if exists(claridge_pickle):
        with open(claridge_pickle, 'rb') as claridge_fp:
            claridge_bundle = pickle.load(claridge_fp)
    else:
        claridge_bundle = get_rdataset('claridge', 'boot')
        with open(claridge_pickle, 'wb') as claridge_fp:
            pickle.dump(claridge_bundle, claridge_fp)
    claridge_data = claridge_bundle.data
    logger.info('claridge data has variables %s' % list(claridge_data))
    claridge_title = claridge_bundle.title
    logger.info('claridge data has title %s' % claridge_title)

    logger.info('loading cloth defect data')
    cloth_pickle = data_folder + 'cloth.pkl'
    if exists(cloth_pickle):
        with open(cloth_pickle, 'rb') as cloth_fp:
            cloth_bundle = pickle.load(cloth_fp)
    else:
        cloth_bundle = get_rdataset('cloth', 'boot')
        with open(cloth_pickle, 'wb') as cloth_fp:
            pickle.dump(cloth_bundle, cloth_fp)
    cloth_data = cloth_bundle.data
    logger.info('cloth data has variables %s' % list(cloth_data))
    cloth_title = cloth_bundle.title
    logger.info('cloth data has title %s' % cloth_title)

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

    logger.info('loading coal mining disaster date data')
    coal_pickle = data_folder + 'coal.pkl'
    if exists(coal_pickle):
        with open(coal_pickle, 'rb') as coal_fp:
            coal_bundle = pickle.load(coal_fp)
    else:
        coal_bundle = get_rdataset('coal', 'boot')
        with open(coal_pickle, 'wb') as coal_fp:
            pickle.dump(coal_bundle, coal_fp)
    coal_data = coal_bundle.data
    logger.info('coal data has variables %s' % list(coal_data))
    coal_title = coal_bundle.title
    logger.info('coal data has title %s' % coal_title)

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

    logger.info('loading Carbon Monoxide transfer data')
    co_transfer_pickle = data_folder + 'co.transfer.pkl'
    if exists(co_transfer_pickle):
        with open(co_transfer_pickle, 'rb') as co_transfer_fp:
            co_transfer_bundle = pickle.load(co_transfer_fp)
    else:
        co_transfer_bundle = get_rdataset('co.transfer', 'boot')
        with open(co_transfer_pickle, 'wb') as co_transfer_fp:
            pickle.dump(co_transfer_bundle, co_transfer_fp)
    co_transfer_data = co_transfer_bundle.data
    logger.info('co.transfer data has variables %s' % list(co_transfer_data))
    co_transfer_title = co_transfer_bundle.title
    logger.info('co.transfer data has title %s' % co_transfer_title)

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

    logger.info("loading Darwin's plant height data")
    darwin_pickle = data_folder + 'darwin.pkl'
    if exists(darwin_pickle):
        with open(darwin_pickle, 'rb') as darwin_fp:
            darwin_bundle = pickle.load(darwin_fp)
    else:
        darwin_bundle = get_rdataset('darwin', 'boot')
        with open(darwin_pickle, 'wb') as darwin_fp:
            pickle.dump(darwin_bundle, darwin_fp)
    darwin_data = darwin_bundle.data
    logger.info('darwin data has variables %s and has %d rows' % (list(darwin_data), len(darwin_data)))
    darwin_title = darwin_bundle.title
    logger.info('darwin data has title %s' % darwin_title)

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

    logger.info('loading domestic dog cardiac data')
    dogs_pickle = data_folder + 'dogs.pkl'
    if exists(dogs_pickle):
        with open(dogs_pickle, 'rb') as dogs_fp:
            dogs_bundle = pickle.load(dogs_fp)
    else:
        dogs_bundle = get_rdataset('dogs', 'boot')
        with open(dogs_pickle, 'wb') as dogs_fp:
            pickle.dump(dogs_bundle, dogs_fp)
    dogs_data = dogs_bundle.data
    logger.info('dogs data has variables %s and has %d rows' % (list(dogs_data), len(dogs_data)))
    dogs_title = dogs_bundle.title
    logger.info('dogs data has title %s' % dogs_title)

    logger.info('loading hybrid duck data')
    ducks_pickle = data_folder + 'ducks.pkl'
    if exists(ducks_pickle):
        with open(ducks_pickle, 'rb') as ducks_fp:
            ducks_bundle = pickle.load(ducks_fp)
    else:
        ducks_bundle = get_rdataset('ducks', 'boot')
        with open(ducks_pickle, 'wb') as ducks_fp:
            pickle.dump(ducks_bundle, ducks_fp)
    ducks_data = ducks_bundle.data
    logger.info('ducks data has variables %s and has %d rows' % (list(ducks_data), len(ducks_data)))
    ducks_title = ducks_bundle.title
    logger.info('ducks data has title %s' % ducks_title)

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

    logger.info('loading balsam fir seedling data')
    fir_pickle = data_folder + 'fir.pkl'
    if exists(fir_pickle):
        with open(fir_pickle, 'rb') as fir_fp:
            fir_bundle = pickle.load(fir_fp)
    else:
        fir_bundle = get_rdataset('fir', 'boot')
        with open(fir_pickle, 'wb') as fir_fp:
            pickle.dump(fir_bundle, fir_fp)
    fir_data = fir_bundle.data
    logger.info('fir data has variables %s and has %d rows' % (list(fir_data), len(fir_data)))
    fir_title = fir_bundle.title
    logger.info('fir data has title %s' % fir_title)

    logger.info('loading fraternal head size data')
    frets_pickle = data_folder + 'frets.pkl'
    if exists(frets_pickle):
        with open(frets_pickle, 'rb') as frets_fp:
            frets_bundle = pickle.load(frets_fp)
    else:
        frets_bundle = get_rdataset('frets', 'boot')
        with open(frets_pickle, 'wb') as frets_fp:
            pickle.dump(frets_bundle, frets_fp)
    frets_data = frets_bundle.data
    logger.info('frets data has variables %s and has %d rows' % (list(frets_data), len(frets_data)))
    frets_title = frets_bundle.title
    logger.info('frets data has title %s' % frets_title)

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

    random_state = 1
    kdd_pickle = data_folder + 'kddcup99.pkl'
    if exists(kdd_pickle):
        with open(kdd_pickle, 'rb') as kdd_fp:
            kdd_bunch = pickle.load(kdd_fp)
    else:
        logger.info('loading KDD data')
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

    logger.info('loading Olivetti faces data')
    olivetti_faces = fetch_olivetti_faces(data_home=data_folder)
    olivetti_faces_data = olivetti_faces['data']
    olivetti_faces_images = olivetti_faces['images']
    olivetti_faces_target = olivetti_faces['target']
    olivetti_faces_description = olivetti_faces['DESCR']

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
    rcv1_bunch = fetch_rcv1(subset='all', download_if_missing=True, random_state=random_state)
    rcv1_data = rcv1_bunch['data']
    rcv1_target = rcv1_bunch['target']
    rcv1_sample_id = rcv1_bunch['sample_id']
    rcv1_target_names = rcv1_bunch['target_names']
    rcv1_description = rcv1_bunch['DESCR']

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

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
