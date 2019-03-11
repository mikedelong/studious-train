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

    logger.info('loading ACF1 data')
    ACF1_pickle = data_folder + 'ACF1.pkl'
    if exists(ACF1_pickle):
        with open(ACF1_pickle, 'rb') as ACF1_fp:
            ACF1_bundle = pickle.load(ACF1_fp)
    else:
        ACF1_bundle = get_rdataset('ACF1', 'DAAG')
        with open(ACF1_pickle, 'wb') as ACF1_fp:
            pickle.dump(ACF1_bundle, ACF1_fp)
    ACF1_data = ACF1_bundle.data
    logger.info('ACF1 data has variables %s' % list(ACF1_data))
    logger.info('ACF1 data has %d rows and %d variables' % ACF1_data.shape)
    ACF1_title = ACF1_bundle.title
    logger.info('ACF1 data has title %s' % ACF1_title)

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

    logger.info('loading affairs data')
    affairs_pickle = data_folder + 'affairs.pkl'
    if exists(affairs_pickle):
        with open(affairs_pickle, 'rb') as affairs_fp:
            affairs_bundle = pickle.load(affairs_fp)
    else:
        affairs_bundle = get_rdataset('affairs', 'COUNT')
        with open(affairs_pickle, 'wb') as affairs_fp:
            pickle.dump(affairs_bundle, affairs_fp)
    affairs_data = affairs_bundle.data
    logger.info('affairs data has variables %s and has %d rows' % (list(affairs_data), len(affairs_data)))
    affairs_title = affairs_bundle.title
    logger.info('affairs data has title %s' % affairs_title)

    logger.info('loading EU agriculture workforce data')
    agriculture_pickle = data_folder + 'agriculture.pkl'
    if exists(agriculture_pickle):
        with open(agriculture_pickle, 'rb') as agriculture_fp:
            agriculture_bundle = pickle.load(agriculture_fp)
    else:
        agriculture_bundle = get_rdataset('agriculture', 'cluster')
        with open(agriculture_pickle, 'wb') as agriculture_fp:
            pickle.dump(agriculture_bundle, agriculture_fp)
    agriculture_data = agriculture_bundle.data
    logger.info('agriculture data has variables %s and has %d rows' % (list(agriculture_data), len(agriculture_data)))
    agriculture_title = agriculture_bundle.title
    logger.info('agriculture data has title %s' % agriculture_title)

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

    logger.info('loading Australian athletic data')
    ais_pickle = data_folder + 'ais.pkl'
    if exists(ais_pickle):
        with open(ais_pickle, 'rb') as ais_fp:
            ais_bundle = pickle.load(ais_fp)
    else:
        ais_bundle = get_rdataset('ais', 'DAAG')
        with open(ais_pickle, 'wb') as ais_fp:
            pickle.dump(ais_bundle, ais_fp)
    ais_data = ais_bundle.data
    logger.info('ais data has variables %s' % list(ais_data))
    logger.info('ais data has %d rows and %d variables' % ais_data.shape)
    ais_title = ais_bundle.title
    logger.info('ais data has title %s' % ais_title)

    logger.info('loading books data')
    allbacks_pickle = data_folder + 'allbacks.pkl'
    if exists(allbacks_pickle):
        with open(allbacks_pickle, 'rb') as allbacks_fp:
            allbacks_bundle = pickle.load(allbacks_fp)
    else:
        allbacks_bundle = get_rdataset('allbacks', 'DAAG')
        with open(allbacks_pickle, 'wb') as allbacks_fp:
            pickle.dump(allbacks_bundle, allbacks_fp)
    allbacks_data = allbacks_bundle.data
    logger.info('allbacks data has variables %s' % list(allbacks_data))
    logger.info('allbacks data has %d rows and %d variables' % allbacks_data.shape)
    allbacks_title = allbacks_bundle.title
    logger.info('allbacks data has title %s' % allbacks_title)

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

    logger.info('loading anesthetic effectiveness data')
    anesthetic_pickle = data_folder + 'anesthetic.pkl'
    if exists(anesthetic_pickle):
        with open(anesthetic_pickle, 'rb') as anesthetic_fp:
            anesthetic_bundle = pickle.load(anesthetic_fp)
    else:
        anesthetic_bundle = get_rdataset('anesthetic', 'DAAG')
        with open(anesthetic_pickle, 'wb') as anesthetic_fp:
            pickle.dump(anesthetic_bundle, anesthetic_fp)
    anesthetic_data = anesthetic_bundle.data
    logger.info('anesthetic data has variables %s' % list(anesthetic_data))
    logger.info('anesthetic data has %d rows and %d variables' % anesthetic_data.shape)
    anesthetic_title = anesthetic_bundle.title
    logger.info('anesthetic data has title %s' % anesthetic_title)

    logger.info('loading animal attribute data')
    animals_pickle = data_folder + 'animals.pkl'
    if exists(animals_pickle):
        with open(animals_pickle, 'rb') as animals_fp:
            animals_bundle = pickle.load(animals_fp)
    else:
        animals_bundle = get_rdataset('animals', 'cluster')
        with open(animals_pickle, 'wb') as animals_fp:
            pickle.dump(animals_bundle, animals_fp)
    animals_data = animals_bundle.data
    logger.info('animals data has variables %s and has %d rows' % (list(animals_data), len(animals_data)))
    animals_title = animals_bundle.title
    logger.info('animals data has title %s' % animals_title)

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

    logger.info('loading corn yields/block 111 data')
    ant111b_pickle = data_folder + 'ant111b.pkl'
    if exists(ant111b_pickle):
        with open(ant111b_pickle, 'rb') as ant111b_fp:
            ant111b_bundle = pickle.load(ant111b_fp)
    else:
        ant111b_bundle = get_rdataset('ant111b', 'DAAG')
        with open(ant111b_pickle, 'wb') as ant111b_fp:
            pickle.dump(ant111b_bundle, ant111b_fp)
    ant111b_data = ant111b_bundle.data
    logger.info('ant111b data has variables %s' % list(ant111b_data))
    logger.info('ant111b data has %d rows and %d variables' % ant111b_data.shape)
    ant111b_title = ant111b_bundle.title
    logger.info('ant111b data has title %s' % ant111b_title)

    logger.info('loading Antigua corn yield data')
    antigua_pickle = data_folder + 'antigua.pkl'
    if exists(antigua_pickle):
        with open(antigua_pickle, 'rb') as antigua_fp:
            antigua_bundle = pickle.load(antigua_fp)
    else:
        antigua_bundle = get_rdataset('antigua', 'DAAG')
        with open(antigua_pickle, 'wb') as antigua_fp:
            pickle.dump(antigua_bundle, antigua_fp)
    antigua_data = antigua_bundle.data
    logger.info('antigua data has variables %s' % list(antigua_data))
    logger.info('antigua data has %d rows and %d variables' % antigua_data.shape)
    antigua_title = antigua_bundle.title
    logger.info('antigua data has title %s' % antigua_title)

    logger.info('loading apple taste data')
    appletaste_pickle = data_folder + 'appletaste.pkl'
    if exists(appletaste_pickle):
        with open(appletaste_pickle, 'rb') as appletaste_fp:
            appletaste_bundle = pickle.load(appletaste_fp)
    else:
        appletaste_bundle = get_rdataset('appletaste', 'DAAG')
        with open(appletaste_pickle, 'wb') as appletaste_fp:
            pickle.dump(appletaste_bundle, appletaste_fp)
    appletaste_data = appletaste_bundle.data
    logger.info('appletaste data has variables %s' % list(appletaste_data))
    logger.info('appletaste data has %d rows and %d variables' % appletaste_data.shape)
    appletaste_title = appletaste_bundle.title
    logger.info('appletaste data has title %s' % appletaste_title)

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

    logger.info('loading Australian lat/lon data')
    aulatlong_pickle = data_folder + 'aulatlong.pkl'
    if exists(aulatlong_pickle):
        with open(aulatlong_pickle, 'rb') as aulatlong_fp:
            aulatlong_bundle = pickle.load(aulatlong_fp)
    else:
        aulatlong_bundle = get_rdataset('aulatlong', 'DAAG')
        with open(aulatlong_pickle, 'wb') as aulatlong_fp:
            pickle.dump(aulatlong_bundle, aulatlong_fp)
    aulatlong_data = aulatlong_bundle.data
    logger.info('aulatlong data has variables %s' % list(aulatlong_data))
    logger.info('aulatlong data has %d rows and %d variables' % aulatlong_data.shape)
    aulatlong_title = aulatlong_bundle.title
    logger.info('aulatlong data has title %s' % aulatlong_title)

    logger.info('loading Australian population data')
    austpop_pickle = data_folder + 'austpop.pkl'
    if exists(austpop_pickle):
        with open(austpop_pickle, 'rb') as austpop_fp:
            austpop_bundle = pickle.load(austpop_fp)
    else:
        austpop_bundle = get_rdataset('austpop', 'DAAG')
        with open(austpop_pickle, 'wb') as austpop_fp:
            pickle.dump(austpop_bundle, austpop_fp)
    austpop_data = austpop_bundle.data
    logger.info('austpop data has variables %s' % list(austpop_data))
    logger.info('austpop data has %d rows and %d variables' % austpop_data.shape)
    austpop_title = austpop_bundle.title
    logger.info('austpop data has title %s' % austpop_title)

    logger.info('loading azcabgptca(?) data')
    azcabgptca_pickle = data_folder + 'azcabgptca.pkl'
    if exists(azcabgptca_pickle):
        with open(azcabgptca_pickle, 'rb') as azcabgptca_fp:
            azcabgptca_bundle = pickle.load(azcabgptca_fp)
    else:
        azcabgptca_bundle = get_rdataset('azcabgptca', 'COUNT')
        with open(azcabgptca_pickle, 'wb') as azcabgptca_fp:
            pickle.dump(azcabgptca_bundle, azcabgptca_fp)
    azcabgptca_data = azcabgptca_bundle.data
    logger.info('azcabgptca data has variables %s and has %d rows' % (list(azcabgptca_data), len(azcabgptca_data)))
    azcabgptca_title = azcabgptca_bundle.title
    logger.info('azcabgptca data has title %s' % azcabgptca_title)

    logger.info('loading azdrg112(?) data')
    azdrg112_pickle = data_folder + 'azdrg112.pkl'
    if exists(azdrg112_pickle):
        with open(azdrg112_pickle, 'rb') as azdrg112_fp:
            azdrg112_bundle = pickle.load(azdrg112_fp)
    else:
        azdrg112_bundle = get_rdataset('azdrg112', 'COUNT')
        with open(azdrg112_pickle, 'wb') as azdrg112_fp:
            pickle.dump(azdrg112_bundle, azdrg112_fp)
    azdrg112_data = azdrg112_bundle.data
    logger.info('azdrg112 data has variables %s and has %d rows' % (list(azdrg112_data), len(azdrg112_data)))
    azdrg112_title = azdrg112_bundle.title
    logger.info('azdrg112 data has title %s' % azdrg112_title)

    logger.info('loading azpro data')
    azpro_pickle = data_folder + 'azpro.pkl'
    if exists(azpro_pickle):
        with open(azpro_pickle, 'rb') as azpro_fp:
            azpro_bundle = pickle.load(azpro_fp)
    else:
        azpro_bundle = get_rdataset('azpro', 'COUNT')
        with open(azpro_pickle, 'wb') as azpro_fp:
            pickle.dump(azpro_bundle, azpro_fp)
    azpro_data = azpro_bundle.data
    logger.info('azpro data has variables %s and has %d rows' % (list(azpro_data), len(azpro_data)))
    azpro_title = azpro_bundle.title
    logger.info('azpro data has title %s' % azpro_title)

    logger.info('loading azprocedure data')
    azprocedure_pickle = data_folder + 'azprocedure.pkl'
    if exists(azprocedure_pickle):
        with open(azprocedure_pickle, 'rb') as azprocedure_fp:
            azprocedure_bundle = pickle.load(azprocedure_fp)
    else:
        azprocedure_bundle = get_rdataset('azprocedure', 'COUNT')
        with open(azprocedure_pickle, 'wb') as azprocedure_fp:
            pickle.dump(azprocedure_bundle, azprocedure_fp)
    azprocedure_data = azprocedure_bundle.data
    logger.info('azprocedure data has variables %s and has %d rows' % (list(azprocedure_data), len(azprocedure_data)))
    azprocedure_title = azprocedure_bundle.title
    logger.info('azprocedure data has title %s' % azprocedure_title)

    logger.info('loading badhealth data')
    badhealth_pickle = data_folder + 'badhealth.pkl'
    if exists(badhealth_pickle):
        with open(badhealth_pickle, 'rb') as badhealth_fp:
            badhealth_bundle = pickle.load(badhealth_fp)
    else:
        badhealth_bundle = get_rdataset('badhealth', 'COUNT')
        with open(badhealth_pickle, 'wb') as badhealth_fp:
            pickle.dump(badhealth_bundle, badhealth_fp)
    badhealth_data = badhealth_bundle.data
    logger.info('badhealth data has variables %s and has %d rows' % (list(badhealth_data), len(badhealth_data)))
    badhealth_title = badhealth_bundle.title
    logger.info('badhealth data has title %s' % badhealth_title)

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

    logger.info('loading biomass data')
    biomass_pickle = data_folder + 'biomass.pkl'
    if exists(biomass_pickle):
        with open(biomass_pickle, 'rb') as biomass_fp:
            biomass_bundle = pickle.load(biomass_fp)
    else:
        biomass_bundle = get_rdataset('biomass', 'DAAG')
        with open(biomass_pickle, 'wb') as biomass_fp:
            pickle.dump(biomass_bundle, biomass_fp)
    biomass_data = biomass_bundle.data
    logger.info('biomass data has variables %s' % list(biomass_data))
    logger.info('biomass data has %d rows and %d variables' % biomass_data.shape)
    biomass_title = biomass_bundle.title
    logger.info('biomass data has title %s' % biomass_title)

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

    logger.info('loading Australian annual climate data')
    bomregions_pickle = data_folder + 'bomregions.pkl'
    if exists(bomregions_pickle):
        with open(bomregions_pickle, 'rb') as bomregions_fp:
            bomregions_bundle = pickle.load(bomregions_fp)
    else:
        bomregions_bundle = get_rdataset('bomregions', 'DAAG')
        with open(bomregions_pickle, 'wb') as bomregions_fp:
            pickle.dump(bomregions_bundle, bomregions_fp)
    bomregions_data = bomregions_bundle.data
    logger.info('bomregions data has variables %s' % list(bomregions_data))
    logger.info('bomregions data has %d rows and %d variables' % bomregions_data.shape)
    bomregions_title = bomregions_bundle.title
    logger.info('bomregions data has title %s' % bomregions_title)

    logger.info('loading Australian regional 2011 annual climate data')
    bomregions2011_pickle = data_folder + 'bomregions2011.pkl'
    if exists(bomregions2011_pickle):
        with open(bomregions2011_pickle, 'rb') as bomregions2011_fp:
            bomregions2011_bundle = pickle.load(bomregions2011_fp)
    else:
        bomregions2011_bundle = get_rdataset('bomregions2011', 'DAAG')
        with open(bomregions2011_pickle, 'wb') as bomregions2011_fp:
            pickle.dump(bomregions2011_bundle, bomregions2011_fp)
    bomregions2011_data = bomregions2011_bundle.data
    logger.info('bomregions2011 data has variables %s' % list(bomregions2011_data))
    logger.info('bomregions2011 data has %d rows and %d variables' % bomregions2011_data.shape)
    bomregions2011_title = bomregions2011_bundle.title
    logger.info('bomregions2011 data has title %s' % bomregions2011_title)

    logger.info('loading Australian regional 2012 annual climate data')
    bomregions2012_pickle = data_folder + 'bomregions2012.pkl'
    if exists(bomregions2012_pickle):
        with open(bomregions2012_pickle, 'rb') as bomregions2012_fp:
            bomregions2012_bundle = pickle.load(bomregions2012_fp)
    else:
        bomregions2012_bundle = get_rdataset('bomregions2012', 'DAAG')
        with open(bomregions2012_pickle, 'wb') as bomregions2012_fp:
            pickle.dump(bomregions2012_bundle, bomregions2012_fp)
    bomregions2012_data = bomregions2012_bundle.data
    logger.info('bomregions2012 data has variables %s' % list(bomregions2012_data))
    logger.info('bomregions2012 data has %d rows and %d variables' % bomregions2012_data.shape)
    bomregions2012_title = bomregions2012_bundle.title
    logger.info('bomregions2012 data has title %s' % bomregions2012_title)

    logger.info('loading Southern Oscillation index data')
    bomsoi_pickle = data_folder + 'bomsoi.pkl'
    if exists(bomsoi_pickle):
        with open(bomsoi_pickle, 'rb') as bomsoi_fp:
            bomsoi_bundle = pickle.load(bomsoi_fp)
    else:
        bomsoi_bundle = get_rdataset('bomsoi', 'DAAG')
        with open(bomsoi_pickle, 'wb') as bomsoi_fp:
            pickle.dump(bomsoi_bundle, bomsoi_fp)
    bomsoi_data = bomsoi_bundle.data
    logger.info('bomsoi data has variables %s' % list(bomsoi_data))
    logger.info('bomsoi data has %d rows and %d variables' % bomsoi_data.shape)
    bomsoi_title = bomsoi_bundle.title
    logger.info('bomsoi data has title %s' % bomsoi_title)

    logger.info('loading Southern Oscillation index 2001 data')
    bomsoi2001_pickle = data_folder + 'bomsoi2001.pkl'
    if exists(bomsoi2001_pickle):
        with open(bomsoi2001_pickle, 'rb') as bomsoi2001_fp:
            bomsoi2001_bundle = pickle.load(bomsoi2001_fp)
    else:
        bomsoi2001_bundle = get_rdataset('bomsoi2001', 'DAAG')
        with open(bomsoi2001_pickle, 'wb') as bomsoi2001_fp:
            pickle.dump(bomsoi2001_bundle, bomsoi2001_fp)
    bomsoi2001_data = bomsoi2001_bundle.data
    logger.info('bomsoi2001 data has variables %s' % list(bomsoi2001_data))
    logger.info('bomsoi2001 data has %d rows and %d variables' % bomsoi2001_data.shape)
    bomsoi2001_title = bomsoi2001_bundle.title
    logger.info('bomsoi2001 data has title %s' % bomsoi2001_title)

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

    logger.info('loading corrected Boston housing data')
    bostonc_pickle = data_folder + 'bostonc.pkl'
    if exists(bostonc_pickle):
        with open(bostonc_pickle, 'rb') as bostonc_fp:
            bostonc_bundle = pickle.load(bostonc_fp)
    else:
        bostonc_bundle = get_rdataset('bostonc', 'DAAG')
        with open(bostonc_pickle, 'wb') as bostonc_fp:
            pickle.dump(bostonc_bundle, bostonc_fp)
    bostonc_data = bostonc_bundle.data
    logger.info('bostonc data has variables %s' % list(bostonc_data))
    logger.info('bostonc data has %d rows and %d variables' % bostonc_data.shape)
    bostonc_title = bostonc_bundle.title
    logger.info('bostonc data has title %s' % bostonc_title)

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

    logger.info('loading Canadian population data')
    CanPop_pickle = data_folder + 'CanPop.pkl'
    if exists(CanPop_pickle):
        with open(CanPop_pickle, 'rb') as CanPop_fp:
            CanPop_bundle = pickle.load(CanPop_fp)
    else:
        CanPop_bundle = get_rdataset('CanPop', 'carData')
        with open(CanPop_pickle, 'wb') as CanPop_fp:
            pickle.dump(CanPop_bundle, CanPop_fp)
    CanPop_data = CanPop_bundle.data
    logger.info('CanPop data has variables %s and has %d rows' % (list(CanPop_data), len(CanPop_data)))
    CanPop_title = CanPop_bundle.title
    logger.info('CanPop data has title %s' % CanPop_title)

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

    logger.info('loading US car price data')
    carprice_pickle = data_folder + 'carprice.pkl'
    if exists(carprice_pickle):
        with open(carprice_pickle, 'rb') as carprice_fp:
            carprice_bundle = pickle.load(carprice_fp)
    else:
        carprice_bundle = get_rdataset('carprice', 'DAAG')
        with open(carprice_pickle, 'wb') as carprice_fp:
            pickle.dump(carprice_bundle, carprice_fp)
    carprice_data = carprice_bundle.data
    logger.info('carprice data has variables %s' % list(carprice_data))
    logger.info('carprice data has %d rows and %d variables' % carprice_data.shape)
    carprice_title = carprice_bundle.title
    logger.info('carprice data has title %s' % carprice_title)

    logger.info('loading Car93 data')
    Cars93_pickle = data_folder + 'Cars93.pkl'
    if exists(Cars93_pickle):
        with open(Cars93_pickle, 'rb') as Cars93_fp:
            Cars93_bundle = pickle.load(Cars93_fp)
    else:
        Cars93_bundle = get_rdataset('Cars93.summary', 'DAAG')
        with open(Cars93_pickle, 'wb') as Cars93_fp:
            pickle.dump(Cars93_bundle, Cars93_fp)
    Cars93_data = Cars93_bundle.data
    logger.info('Cars93 data has variables %s' % list(Cars93_data))
    logger.info('Cars93 data has %d rows and %d variables' % Cars93_data.shape)
    Cars93_title = Cars93_bundle.title
    logger.info('Cars93 data has title %s' % Cars93_title)

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

    logger.info('loading breakfast cereal sugar data')
    cerealsugar_pickle = data_folder + 'cerealsugar.pkl'
    if exists(cerealsugar_pickle):
        with open(cerealsugar_pickle, 'rb') as cerealsugar_fp:
            cerealsugar_bundle = pickle.load(cerealsugar_fp)
    else:
        cerealsugar_bundle = get_rdataset('cerealsugar', 'DAAG')
        with open(cerealsugar_pickle, 'wb') as cerealsugar_fp:
            pickle.dump(cerealsugar_bundle, cerealsugar_fp)
    cerealsugar_data = cerealsugar_bundle.data
    logger.info('cerealsugar data has variables %s' % list(cerealsugar_data))
    logger.info('cerealsugar data has %d rows and %d variables' % cerealsugar_data.shape)
    cerealsugar_title = cerealsugar_bundle.title
    logger.info('cerealsugar data has title %s' % cerealsugar_title)

    logger.info('loading Cape fur seal data')
    cfseal_pickle = data_folder + 'cfseal.pkl'
    if exists(cfseal_pickle):
        with open(cfseal_pickle, 'rb') as cfseal_fp:
            cfseal_bundle = pickle.load(cfseal_fp)
    else:
        cfseal_bundle = get_rdataset('cfseal', 'DAAG')
        with open(cfseal_pickle, 'wb') as cfseal_fp:
            pickle.dump(cfseal_bundle, cfseal_fp)
    cfseal_data = cfseal_bundle.data
    logger.info('cfseal data has variables %s' % list(cfseal_data))
    logger.info('cfseal data has %d rows and %d variables' % cfseal_data.shape)
    cfseal_title = cfseal_bundle.title
    logger.info('cfseal data has title %s' % cfseal_title)

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

    logger.info('loading Chilean voting data')
    Chile_pickle = data_folder + 'Chile.pkl'
    if exists(Chile_pickle):
        with open(Chile_pickle, 'rb') as Chile_fp:
            Chile_bundle = pickle.load(Chile_fp)
    else:
        Chile_bundle = get_rdataset('Chile', 'carData')
        with open(Chile_pickle, 'wb') as Chile_fp:
            pickle.dump(Chile_bundle, Chile_fp)
    Chile_data = Chile_bundle.data
    logger.info('Chile data has variables %s and has %d rows' % (list(Chile_data), len(Chile_data)))
    Chile_title = Chile_bundle.title
    logger.info('Chile data has title %s' % Chile_title)

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

    logger.info('loading Romanian peasant revolt data')
    Chirot_pickle = data_folder + 'Chirot.pkl'
    if exists(Chirot_pickle):
        with open(Chirot_pickle, 'rb') as Chirot_fp:
            Chirot_bundle = pickle.load(Chirot_fp)
    else:
        Chirot_bundle = get_rdataset('Chirot', 'carData')
        with open(Chirot_pickle, 'wb') as Chirot_fp:
            pickle.dump(Chirot_bundle, Chirot_fp)
    Chirot_data = Chirot_bundle.data
    logger.info('Chirot data has variables %s and has %d rows' % (list(Chirot_data), len(Chirot_data)))
    Chirot_title = Chirot_bundle.title
    logger.info('Chirot data has title %s' % Chirot_title)

    logger.info('loading kola data')
    chorSub_pickle = data_folder + 'chorSub.pkl'
    if exists(chorSub_pickle):
        with open(chorSub_pickle, 'rb') as chorSub_fp:
            chorSub_bundle = pickle.load(chorSub_fp)
    else:
        chorSub_bundle = get_rdataset('chorSub', 'cluster')
        with open(chorSub_pickle, 'wb') as chorSub_fp:
            pickle.dump(chorSub_bundle, chorSub_fp)
    chorSub_data = chorSub_bundle.data
    logger.info('chorSub data has variables %s and has %d rows' % (list(chorSub_data), len(chorSub_data)))
    chorSub_title = chorSub_bundle.title
    logger.info('chorSub data has title %s' % chorSub_title)

    logger.info('loading Canadian major city population data')
    cities_pickle = data_folder + 'cities.pkl'
    if exists(cities_pickle):
        with open(cities_pickle, 'rb') as cities_fp:
            cities_bundle = pickle.load(cities_fp)
    else:
        cities_bundle = get_rdataset('cities', 'DAAG')
        with open(cities_pickle, 'wb') as cities_fp:
            pickle.dump(cities_bundle, cities_fp)
    cities_data = cities_bundle.data
    logger.info('cities data has variables %s' % list(cities_data))
    logger.info('cities data has %d rows and %d variables' % cities_data.shape)
    cities_title = cities_bundle.title
    logger.info('cities data has title %s' % cities_title)

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

    logger.info('loading codling moth dose data')
    codling_pickle = data_folder + 'codling.pkl'
    if exists(codling_pickle):
        with open(codling_pickle, 'rb') as codling_fp:
            codling_bundle = pickle.load(codling_fp)
    else:
        codling_bundle = get_rdataset('codling', 'DAAG')
        with open(codling_pickle, 'wb') as codling_fp:
            pickle.dump(codling_bundle, codling_fp)
    codling_data = codling_bundle.data
    logger.info('codling data has variables %s' % list(codling_data))
    logger.info('codling data has %d rows and %d variables' % codling_data.shape)
    codling_title = codling_bundle.title
    logger.info('codling data has title %s' % codling_title)

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

    logger.info('loading British cotton workers data')
    cottonworkers_pickle = data_folder + 'cottonworkers.pkl'
    if exists(cottonworkers_pickle):
        with open(cottonworkers_pickle, 'rb') as cottonworkers_fp:
            cottonworkers_bundle = pickle.load(cottonworkers_fp)
    else:
        cottonworkers_bundle = get_rdataset('cottonworkers', 'DAAG')
        with open(cottonworkers_pickle, 'wb') as cottonworkers_fp:
            pickle.dump(cottonworkers_bundle, cottonworkers_fp)
    cottonworkers_data = cottonworkers_bundle.data
    logger.info('cottonworkers data has variables %s' % list(cottonworkers_data))
    logger.info('cottonworkers data has %d rows and %d variables' % cottonworkers_data.shape)
    cottonworkers_title = cottonworkers_bundle.title
    logger.info('cottonworkers data has title %s' % cottonworkers_title)

    logger.info('loading volunteering data')
    Cowles_pickle = data_folder + 'Cowles.pkl'
    if exists(Cowles_pickle):
        with open(Cowles_pickle, 'rb') as Cowles_fp:
            Cowles_bundle = pickle.load(Cowles_fp)
    else:
        Cowles_bundle = get_rdataset('Cowles', 'carData')
        with open(Cowles_pickle, 'wb') as Cowles_fp:
            pickle.dump(Cowles_bundle, Cowles_fp)
    Cowles_data = Cowles_bundle.data
    logger.info('Cowles data has variables %s and has %d rows' % (list(Cowles_data), len(Cowles_data)))
    Cowles_title = Cowles_bundle.title
    logger.info('Cowles data has title %s' % Cowles_title)

    logger.info('loading labor training evaluation data')
    cps1_pickle = data_folder + 'cps1.pkl'
    if exists(cps1_pickle):
        with open(cps1_pickle, 'rb') as cps1_fp:
            cps1_bundle = pickle.load(cps1_fp)
    else:
        cps1_bundle = get_rdataset('cps1', 'DAAG')
        with open(cps1_pickle, 'wb') as cps1_fp:
            pickle.dump(cps1_bundle, cps1_fp)
    cps1_data = cps1_bundle.data
    logger.info('cps1 data has variables %s' % list(cps1_data))
    logger.info('cps1 data has %d rows and %d variables' % cps1_data.shape)
    cps1_title = cps1_bundle.title
    logger.info('cps1 data has title %s' % cps1_title)

    logger.info('loading labor training evaluation data')
    cps1_pickle = data_folder + 'cps1.pkl'
    if exists(cps1_pickle):
        with open(cps1_pickle, 'rb') as cps1_fp:
            cps1_bundle = pickle.load(cps1_fp)
    else:
        cps1_bundle = get_rdataset('cps1', 'DAAG')
        with open(cps1_pickle, 'wb') as cps1_fp:
            pickle.dump(cps1_bundle, cps1_fp)
    cps1_data = cps1_bundle.data
    logger.info('cps1 data has variables %s' % list(cps1_data))
    logger.info('cps1 data has %d rows and %d variables' % cps1_data.shape)
    cps1_title = cps1_bundle.title
    logger.info('cps1 data has title %s' % cps1_title)

    logger.info('loading labor training evaluation data')
    cps2_pickle = data_folder + 'cps2.pkl'
    if exists(cps2_pickle):
        with open(cps2_pickle, 'rb') as cps2_fp:
            cps2_bundle = pickle.load(cps2_fp)
    else:
        cps2_bundle = get_rdataset('cps2', 'DAAG')
        with open(cps2_pickle, 'wb') as cps2_fp:
            pickle.dump(cps2_bundle, cps2_fp)
    cps2_data = cps2_bundle.data
    logger.info('cps2 data has variables %s' % list(cps2_data))
    logger.info('cps2 data has %d rows and %d variables' % cps2_data.shape)
    cps2_title = cps2_bundle.title
    logger.info('cps2 data has title %s' % cps2_title)

    logger.info('loading labor training evaluation data')
    cps3_pickle = data_folder + 'cps3.pkl'
    if exists(cps3_pickle):
        with open(cps3_pickle, 'rb') as cps3_fp:
            cps3_bundle = pickle.load(cps3_fp)
    else:
        cps3_bundle = get_rdataset('cps3', 'DAAG')
        with open(cps3_pickle, 'wb') as cps3_fp:
            pickle.dump(cps3_bundle, cps3_fp)
    cps3_data = cps3_bundle.data
    logger.info('cps3 data has variables %s' % list(cps3_data))
    logger.info('cps3 data has %d rows and %d variables' % cps3_data.shape)
    cps3_title = cps3_bundle.title
    logger.info('cps3 data has title %s' % cps3_title)

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

    logger.info('loading cricketer lifespan data')
    cricketer_pickle = data_folder + 'cricketer.pkl'
    if exists(cricketer_pickle):
        with open(cricketer_pickle, 'rb') as cricketer_fp:
            cricketer_bundle = pickle.load(cricketer_fp)
    else:
        cricketer_bundle = get_rdataset('cricketer', 'DAAG')
        with open(cricketer_pickle, 'wb') as cricketer_fp:
            pickle.dump(cricketer_bundle, cricketer_fp)
    cricketer_data = cricketer_bundle.data
    logger.info('cricketer data has variables %s' % list(cricketer_data))
    logger.info('cricketer data has %d rows and %d variables' % cricketer_data.shape)
    cricketer_title = cricketer_bundle.title
    logger.info('cricketer data has title %s' % cricketer_title)

    logger.info('loading cuckoo vs. host egg size data')
    cuckoohosts_pickle = data_folder + 'cuckoohosts.pkl'
    if exists(cuckoohosts_pickle):
        with open(cuckoohosts_pickle, 'rb') as cuckoohosts_fp:
            cuckoohosts_bundle = pickle.load(cuckoohosts_fp)
    else:
        cuckoohosts_bundle = get_rdataset('cuckoohosts', 'DAAG')
        with open(cuckoohosts_pickle, 'wb') as cuckoohosts_fp:
            pickle.dump(cuckoohosts_bundle, cuckoohosts_fp)
    cuckoohosts_data = cuckoohosts_bundle.data
    logger.info('cuckoohosts data has variables %s' % list(cuckoohosts_data))
    logger.info('cuckoohosts data has %d rows and %d variables' % cuckoohosts_data.shape)
    cuckoohosts_title = cuckoohosts_bundle.title
    logger.info('cuckoohosts data has title %s' % cuckoohosts_title)

    logger.info('loading cuckoo egg data')
    cuckoos_pickle = data_folder + 'cuckoos.pkl'
    if exists(cuckoos_pickle):
        with open(cuckoos_pickle, 'rb') as cuckoos_fp:
            cuckoos_bundle = pickle.load(cuckoos_fp)
    else:
        cuckoos_bundle = get_rdataset('cuckoos', 'DAAG')
        with open(cuckoos_pickle, 'wb') as cuckoos_fp:
            pickle.dump(cuckoos_bundle, cuckoos_fp)
    cuckoos_data = cuckoos_bundle.data
    logger.info('cuckoos data has variables %s' % list(cuckoos_data))
    logger.info('cuckoos data has %d rows and %d variables' % cuckoos_data.shape)
    cuckoos_title = cuckoos_bundle.title
    logger.info('cuckoos data has title %s' % cuckoos_title)

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

    logger.info('loading self-reported height and weight data')
    Davis_pickle = data_folder + 'Davis.pkl'
    if exists(Davis_pickle):
        with open(Davis_pickle, 'rb') as Davis_fp:
            Davis_bundle = pickle.load(Davis_fp)
    else:
        Davis_bundle = get_rdataset('Davis', 'carData')
        with open(Davis_pickle, 'wb') as Davis_fp:
            pickle.dump(Davis_bundle, Davis_fp)
    Davis_data = Davis_bundle.data
    logger.info('Davis data has variables %s and has %d rows' % (list(Davis_data), len(Davis_data)))
    Davis_title = Davis_bundle.title
    logger.info('Davis data has title %s' % Davis_title)

    logger.info('loading Drive for Thinness data')
    DavisThin_pickle = data_folder + 'DavisThin.pkl'
    if exists(DavisThin_pickle):
        with open(DavisThin_pickle, 'rb') as DavisThin_fp:
            DavisThin_bundle = pickle.load(DavisThin_fp)
    else:
        DavisThin_bundle = get_rdataset('DavisThin', 'carData')
        with open(DavisThin_pickle, 'wb') as DavisThin_fp:
            pickle.dump(DavisThin_bundle, DavisThin_fp)
    DavisThin_data = DavisThin_bundle.data
    logger.info('DavisThin data has variables %s and has %d rows' % (list(DavisThin_data), len(DavisThin_data)))
    DavisThin_title = DavisThin_bundle.title
    logger.info('DavisThin data has title %s' % DavisThin_title)

    logger.info('loading regional dengue fever data')
    dengue_pickle = data_folder + 'dengue.pkl'
    if exists(dengue_pickle):
        with open(dengue_pickle, 'rb') as dengue_fp:
            dengue_bundle = pickle.load(dengue_fp)
    else:
        dengue_bundle = get_rdataset('dengue', 'DAAG')
        with open(dengue_pickle, 'wb') as dengue_fp:
            pickle.dump(dengue_bundle, dengue_fp)
    dengue_data = dengue_bundle.data
    logger.info('dengue data has variables %s' % list(dengue_data))
    logger.info('dengue data has %d rows and %d variables' % dengue_data.shape)
    dengue_title = dengue_bundle.title
    logger.info('dengue data has title %s' % dengue_title)

    logger.info('loading Minnesota wolf data')
    Depredations_pickle = data_folder + 'Depredations.pkl'
    if exists(Depredations_pickle):
        with open(Depredations_pickle, 'rb') as Depredations_fp:
            Depredations_bundle = pickle.load(Depredations_fp)
    else:
        Depredations_bundle = get_rdataset('Depredations', 'carData')
        with open(Depredations_pickle, 'wb') as Depredations_fp:
            pickle.dump(Depredations_bundle, Depredations_fp)
    Depredations_data = Depredations_bundle.data
    logger.info(
        'Depredations data has variables %s and has %d rows' % (list(Depredations_data), len(Depredations_data)))
    Depredations_title = Depredations_bundle.title
    logger.info('Depredations data has title %s' % Depredations_title)

    logger.info('loading dewpoint data')
    dewpoint_pickle = data_folder + 'dewpoint.pkl'
    if exists(dewpoint_pickle):
        with open(dewpoint_pickle, 'rb') as dewpoint_fp:
            dewpoint_bundle = pickle.load(dewpoint_fp)
    else:
        dewpoint_bundle = get_rdataset('dewpoint', 'DAAG')
        with open(dewpoint_pickle, 'wb') as dewpoint_fp:
            pickle.dump(dewpoint_bundle, dewpoint_fp)
    dewpoint_data = dewpoint_bundle.data
    logger.info('dewpoint data has variables %s' % list(dewpoint_data))
    logger.info('dewpoint data has %d rows and %d variables' % dewpoint_data.shape)
    dewpoint_title = dewpoint_bundle.title
    logger.info('dewpoint data has title %s' % dewpoint_title)

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

    logger.info('loading droughts data')
    droughts_pickle = data_folder + 'droughts.pkl'
    if exists(droughts_pickle):
        with open(droughts_pickle, 'rb') as droughts_fp:
            droughts_bundle = pickle.load(droughts_fp)
    else:
        droughts_bundle = get_rdataset('droughts', 'DAAG')
        with open(droughts_pickle, 'wb') as droughts_fp:
            pickle.dump(droughts_bundle, droughts_fp)
    droughts_data = droughts_bundle.data
    logger.info('droughts data has variables %s' % list(droughts_data))
    logger.info('droughts data has %d rows and %d variables' % droughts_data.shape)
    droughts_title = droughts_bundle.title
    logger.info('droughts data has title %s' % droughts_title)

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

    logger.info('loading work prestige data')
    Duncan_pickle = data_folder + 'Duncan.pkl'
    if exists(Duncan_pickle):
        with open(Duncan_pickle, 'rb') as Duncan_fp:
            Duncan_bundle = pickle.load(Duncan_fp)
    else:
        Duncan_bundle = get_rdataset('Duncan', 'carData')
        with open(Duncan_pickle, 'wb') as Duncan_fp:
            pickle.dump(Duncan_bundle, Duncan_fp)
    Duncan_data = Duncan_bundle.data
    logger.info(
        'Duncan data has variables %s and has %d rows' % (list(Duncan_data), len(Duncan_data)))
    Duncan_title = Duncan_bundle.title
    logger.info('Duncan data has title %s' % Duncan_title)

    logger.info('loading EPICA CO2 data')
    edcCO2_pickle = data_folder + 'edcCO2.pkl'
    if exists(edcCO2_pickle):
        with open(edcCO2_pickle, 'rb') as edcCO2_fp:
            edcCO2_bundle = pickle.load(edcCO2_fp)
    else:
        edcCO2_bundle = get_rdataset('edcCO2', 'DAAG')
        with open(edcCO2_pickle, 'wb') as edcCO2_fp:
            pickle.dump(edcCO2_bundle, edcCO2_fp)
    edcCO2_data = edcCO2_bundle.data
    logger.info('edcCO2 data has variables %s' % list(edcCO2_data))
    logger.info('edcCO2 data has %d rows and %d variables' % edcCO2_data.shape)
    edcCO2_title = edcCO2_bundle.title
    logger.info('edcCO2 data has title %s' % edcCO2_title)

    logger.info('loading EPICA temperature data')
    edcT_pickle = data_folder + 'edcT.pkl'
    if exists(edcT_pickle):
        with open(edcT_pickle, 'rb') as edcT_fp:
            edcT_bundle = pickle.load(edcT_fp)
    else:
        edcT_bundle = get_rdataset('edcT', 'DAAG')
        with open(edcT_pickle, 'wb') as edcT_fp:
            pickle.dump(edcT_bundle, edcT_fp)
    edcT_data = edcT_bundle.data
    logger.info('edcT data has variables %s' % list(edcT_data))
    logger.info('edcT data has %d rows and %d variables' % edcT_data.shape)
    edcT_title = edcT_bundle.title
    logger.info('edcT data has title %s' % edcT_title)

    logger.info('loading elastic band data')
    elastic1_pickle = data_folder + 'elastic1.pkl'
    if exists(elastic1_pickle):
        with open(elastic1_pickle, 'rb') as elastic1_fp:
            elastic1_bundle = pickle.load(elastic1_fp)
    else:
        elastic1_bundle = get_rdataset('elastic1', 'DAAG')
        with open(elastic1_pickle, 'wb') as elastic1_fp:
            pickle.dump(elastic1_bundle, elastic1_fp)
    elastic1_data = elastic1_bundle.data
    logger.info('elastic1 data has variables %s' % list(elastic1_data))
    logger.info('elastic1 data has %d rows and %d variables' % elastic1_data.shape)
    elastic1_title = elastic1_bundle.title
    logger.info('elastic1 data has title %s' % elastic1_title)

    logger.info('loading elastic band data')
    elastic2_pickle = data_folder + 'elastic2.pkl'
    if exists(elastic2_pickle):
        with open(elastic2_pickle, 'rb') as elastic2_fp:
            elastic2_bundle = pickle.load(elastic2_fp)
    else:
        elastic2_bundle = get_rdataset('elastic2', 'DAAG')
        with open(elastic2_pickle, 'wb') as elastic2_fp:
            pickle.dump(elastic2_bundle, elastic2_fp)
    elastic2_data = elastic2_bundle.data
    logger.info('elastic2 data has variables %s' % list(elastic2_data))
    logger.info('elastic2 data has %d rows and %d variables' % elastic2_data.shape)
    elastic2_title = elastic2_bundle.title
    logger.info('elastic2 data has title %s' % elastic2_title)

    logger.info('loading elastic band data')
    elasticband_pickle = data_folder + 'elasticband.pkl'
    if exists(elasticband_pickle):
        with open(elasticband_pickle, 'rb') as elasticband_fp:
            elasticband_bundle = pickle.load(elasticband_fp)
    else:
        elasticband_bundle = get_rdataset('elasticband', 'DAAG')
        with open(elasticband_pickle, 'wb') as elasticband_fp:
            pickle.dump(elasticband_bundle, elasticband_fp)
    elasticband_data = elasticband_bundle.data
    logger.info('elasticband data has variables %s' % list(elasticband_data))
    logger.info('elasticband data has %d rows and %d variables' % elasticband_data.shape)
    elasticband_title = elasticband_bundle.title
    logger.info('elasticband data has title %s' % elasticband_title)

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

    logger.info('loading census under-count data')
    Ericksen_pickle = data_folder + 'Ericksen.pkl'
    if exists(Ericksen_pickle):
        with open(Ericksen_pickle, 'rb') as Ericksen_fp:
            Ericksen_bundle = pickle.load(Ericksen_fp)
    else:
        Ericksen_bundle = get_rdataset('Ericksen', 'carData')
        with open(Ericksen_pickle, 'wb') as Ericksen_fp:
            pickle.dump(Ericksen_bundle, Ericksen_fp)
    Ericksen_data = Ericksen_bundle.data
    logger.info(
        'Ericksen data has variables %s and has %d rows' % (list(Ericksen_data), len(Ericksen_data)))
    Ericksen_title = Ericksen_bundle.title
    logger.info('Ericksen data has title %s' % Ericksen_title)

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

    logger.info('loading fasttrakg data')
    fasttrakg_pickle = data_folder + 'fasttrakg.pkl'
    if exists(fasttrakg_pickle):
        with open(fasttrakg_pickle, 'rb') as fasttrakg_fp:
            fasttrakg_bundle = pickle.load(fasttrakg_fp)
    else:
        fasttrakg_bundle = get_rdataset('fasttrakg', 'COUNT')
        with open(fasttrakg_pickle, 'wb') as fasttrakg_fp:
            pickle.dump(fasttrakg_bundle, fasttrakg_fp)
    fasttrakg_data = fasttrakg_bundle.data
    logger.info('fasttrakg data has variables %s and has %d rows' % (list(fasttrakg_data), len(fasttrakg_data)))
    fasttrakg_title = fasttrakg_bundle.title
    logger.info('fasttrakg data has title %s' % fasttrakg_title)

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

    logger.info('loading fishing data')
    fishing_pickle = data_folder + 'fishing.pkl'
    if exists(fishing_pickle):
        with open(fishing_pickle, 'rb') as fishing_fp:
            fishing_bundle = pickle.load(fishing_fp)
    else:
        fishing_bundle = get_rdataset('fishing', 'COUNT')
        with open(fishing_pickle, 'wb') as fishing_fp:
            pickle.dump(fishing_bundle, fishing_fp)
    fishing_data = fishing_bundle.data
    logger.info('fishing data has variables %s and has %d rows' % (list(fishing_data), len(fishing_data)))
    fishing_title = fishing_bundle.title
    logger.info('fishing data has title %s' % fishing_title)

    logger.info('loading Florida voting data')
    Florida_pickle = data_folder + 'Florida.pkl'
    if exists(Florida_pickle):
        with open(Florida_pickle, 'rb') as Florida_fp:
            Florida_bundle = pickle.load(Florida_fp)
    else:
        Florida_bundle = get_rdataset('Florida', 'carData')
        with open(Florida_pickle, 'wb') as Florida_fp:
            pickle.dump(Florida_bundle, Florida_fp)
    Florida_data = Florida_bundle.data
    logger.info(
        'Florida data has variables %s and has %d rows' % (list(Florida_data), len(Florida_data)))
    Florida_title = Florida_bundle.title
    logger.info('Florida data has title %s' % Florida_title)

    logger.info('loading flower characteristics data')
    flower_pickle = data_folder + 'flower.pkl'
    if exists(flower_pickle):
        with open(flower_pickle, 'rb') as flower_fp:
            flower_bundle = pickle.load(flower_fp)
    else:
        flower_bundle = get_rdataset('flower', 'cluster')
        with open(flower_pickle, 'wb') as flower_fp:
            pickle.dump(flower_bundle, flower_fp)
    flower_data = flower_bundle.data
    logger.info('flower data has variables %s and has %d rows' % (list(flower_data), len(flower_data)))
    flower_title = flower_bundle.title
    logger.info('flower data has title %s' % flower_title)

    logger.info('loading fossil fuel data')
    fossilfuel_pickle = data_folder + 'fossilfuel.pkl'
    if exists(fossilfuel_pickle):
        with open(fossilfuel_pickle, 'rb') as fossilfuel_fp:
            fossilfuel_bundle = pickle.load(fossilfuel_fp)
    else:
        fossilfuel_bundle = get_rdataset('fossilfuel', 'DAAG')
        with open(fossilfuel_pickle, 'wb') as fossilfuel_fp:
            pickle.dump(fossilfuel_bundle, fossilfuel_fp)
    fossilfuel_data = fossilfuel_bundle.data
    logger.info('fossilfuel data has variables %s' % list(fossilfuel_data))
    logger.info('fossilfuel data has %d rows and %d variables' % fossilfuel_data.shape)
    fossilfuel_title = fossilfuel_bundle.title
    logger.info('fossilfuel data has title %s' % fossilfuel_title)

    logger.info('loading female possum data')
    fossum_pickle = data_folder + 'fossum.pkl'
    if exists(fossum_pickle):
        with open(fossum_pickle, 'rb') as fossum_fp:
            fossum_bundle = pickle.load(fossum_fp)
    else:
        fossum_bundle = get_rdataset('fossum', 'DAAG')
        with open(fossum_pickle, 'wb') as fossum_fp:
            pickle.dump(fossum_bundle, fossum_fp)
    fossum_data = fossum_bundle.data
    logger.info('fossum data has variables %s' % list(fossum_data))
    logger.info('fossum data has %d rows and %d variables' % fossum_data.shape)
    fossum_title = fossum_bundle.title
    logger.info('fossum data has title %s' % fossum_title)

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

    logger.info('loading Crowding/crime in US Metro areas data')
    Freedman_pickle = data_folder + 'Freedman.pkl'
    if exists(Freedman_pickle):
        with open(Freedman_pickle, 'rb') as Freedman_fp:
            Freedman_bundle = pickle.load(Freedman_fp)
    else:
        Freedman_bundle = get_rdataset('Freedman', 'carData')
        with open(Freedman_pickle, 'wb') as Freedman_fp:
            pickle.dump(Freedman_bundle, Freedman_fp)
    Freedman_data = Freedman_bundle.data
    logger.info(
        'Freedman data has variables %s and has %d rows' % (list(Freedman_data), len(Freedman_data)))
    Freedman_title = Freedman_bundle.title
    logger.info('Freedman data has title %s' % Freedman_title)

    logger.info('loading Format effects on recall data')
    Friendly_pickle = data_folder + 'Friendly.pkl'
    if exists(Friendly_pickle):
        with open(Friendly_pickle, 'rb') as Friendly_fp:
            Friendly_bundle = pickle.load(Friendly_fp)
    else:
        Friendly_bundle = get_rdataset('Friendly', 'carData')
        with open(Friendly_pickle, 'wb') as Friendly_fp:
            pickle.dump(Friendly_bundle, Friendly_fp)
    Friendly_data = Friendly_bundle.data
    logger.info(
        'Friendly data has variables %s and has %d rows' % (list(Friendly_data), len(Friendly_data)))
    Friendly_title = Friendly_bundle.title
    logger.info('Friendly data has title %s' % Friendly_title)

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
