# https://www.statsmodels.org/dev/datasets/index.html
import logging
import pickle
from os.path import exists
from time import time

import pandas  as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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
    logger.info('names are %s' % anes96_names)
    anes96_endog = anes96_bunch['endog_name']
    logger.info('ANES96 endogengous variable is %s' % anes96_endog)
    anes96_exog = anes96_bunch['exog_name']
    logger.info('ANES96 exogengous variable is %s' % anes96_exog)

    random_state = 1

    data_df = anes96_data.copy(deep=True)
    features = [item for item in list(anes96_data) if item != 'logpopul']
    for feature in features:
        encoder = LabelEncoder()
        data_df[feature] = encoder.fit_transform(data_df[feature])

    data_df = pd.get_dummies(data_df, columns=['vote', 'educ', 'income', 'selfLR', 'ClinLR', 'DoleLR', 'TVnews', 'age'])
    logger.info(list(data_df))
    features = [item for item in list(data_df) if item not in anes96_endog]
    X_train, X_test, y_train, y_test = train_test_split(data_df[features], data_df[anes96_endog],
                                                        test_size=0.33, random_state=random_state)
    for criterion in ['entropy', 'gini']:
        model = DecisionTreeClassifier(criterion=criterion, splitter='best', max_depth=None, min_samples_split=2,
                                       min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
                                       random_state=random_state, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                       min_impurity_split=None, class_weight=None, presort=False)
        model.fit(X=X_train, y=y_train.values)
        logger.info(
            'feature importance: %s' % {features[i]: model.feature_importances_[i] for i in range(len(features))})
        y_predicted = model.predict(X=X_test)
        logger.info('criterion: %s weather confusion matrix: \n%s' % (
            criterion, confusion_matrix(y_true=y_test, y_pred=y_predicted)))

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
