# https://www.statsmodels.org/dev/datasets/index.html
import logging
import pickle
from os.path import exists
from time import time

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from statsmodels.datasets import ccard

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
    logger.info('ccard data is %d x %d' % ccard_data.shape)
    ccard_names = ccard_bunch['names']
    logger.info('names are %s' % ccard_names)
    ccard_endog = ccard_bunch['endog_name']
    logger.info('ccard endogengous variable is %s' % ccard_endog)
    ccard_exog = ccard_bunch['exog_name']
    logger.info('ccard exogengous variable is %s' % ccard_exog)

    random_state = 1

    data_df = ccard_data.copy(deep=True)

    logger.info(list(data_df))
    features = [item for item in list(data_df) if item not in ccard_endog]
    X_train, X_test, y_train, y_test = train_test_split(data_df[features], data_df[ccard_endog],
                                                        test_size=0.33, random_state=random_state)
    for criterion in ['mse']:
        model = DecisionTreeRegressor(
            criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
            min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None, presort=False)
        model.fit(X=X_train, y=y_train.values)
        logger.info(
            'feature importance: %s' % {features[i]: model.feature_importances_[i] for i in range(len(features))})
        y_predicted = model.predict(X=X_test)
        logger.info('criterion: %s ccard PID accuracy score: %.3f' % (criterion, model.score(X=X_test, y=y_test)))

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
