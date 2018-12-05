import logging
from string import punctuation
from time import time

import pandas as pd
from plotly.graph_objs import Scatter
from plotly.offline import plot
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def count_punctuation(arg):
    if type(arg) is not str:
        return 0
    else:
        return sum([1 for character in arg if character in punctuation])


def modify_author(arg):
    if ',' not in arg:
        return arg
    else:
        (piece0, piece1) = arg.split(',')
        return ' '.join([piece1, piece0]).strip()


if __name__ == '__main__':
    start_time = time()

    console_formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.info('started')

    # our data originally came from here https://gist.github.com/jaidevd but has been cleaned up somewhat

    usecols = ['Title', 'Author']
    source_df = pd.read_csv('../data/books_clean.csv', usecols=usecols, sep=';')
    logger.info('initially source data has %d rows' % len(source_df))

    names = source_df['Author'].values
    bogeys = source_df['Title'].values
    # build the bigrams list; this is our feature list
    bigrams = sorted(
        list(set([item[i:i + 2] for item in names.tolist() + bogeys.tolist() for i in range(len(item) - 1)])))
    logger.info('we found %d bigrams' % len(bigrams))
    data_dict = dict()
    for item in set(names.tolist() + bogeys.tolist()):
        logger.info('%s %s' % (item, item in names))
        data_dict[item] = [int(bigram in item) for bigram in bigrams] + [int(item in names)]

    data_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=bigrams + ['target'])

    # now let's add the string length feature
    data_df['len'] = data_df.index.str.len()

    # now let's add the punctuation count
    data_df['punctuation_count'] = data_df.index.map(count_punctuation)

    logger.info('our data has %d rows and %d columns' % data_df.shape)
    data_file = '../output/bigrams_data.csv'
    data_df.to_csv(data_file)

    test_size = 0.33
    split_random_state = 1
    features = [item for item in list(data_df) if item != 'target']
    X_train, X_test, y_train, y_test = train_test_split(data_df[features], data_df['target'].values, shuffle=True,
                                                        test_size=test_size, random_state=split_random_state)

    current = None
    current_importance = None
    for index, random_state in enumerate(range(1000)):
        model = DecisionTreeClassifier(random_state=random_state)
        model.fit(X=X_train, y=y_train)
        y_predicted = model.predict(X=X_test)
        # use a weighted average to update our current prediction
        current = y_predicted if index == 0 else (float(index) * current + y_predicted) * (1.0 / float(index + 1))
        current_importance = model.feature_importances_ if index == 0 else (float(
            index) * current_importance + model.feature_importances_) * (1.0 / float(index + 1))
        if index % 100 == 0:
            logger.info('%d names: %s \nexpected: %s \nactual: %s \ncurrent: %s' %
                        (random_state, X_test.index.values, y_test, y_predicted, current))

    current_hard = (current >= 0.5).astype('float')
    logger.info('f1: %.3f confusion matrix: \n%s' % (f1_score(y_true=y_test, y_pred=current_hard),
                                                     confusion_matrix(y_true=y_test, y_pred=current_hard)))
    plot([Scatter(x=y_test, y=current, text=X_test.index.values, mode='markers+text')], auto_open=False,
         show_link=False, filename='../output/bigrams.html')

    importance_dict = dict()
    count = 0
    for index, value in enumerate(current_importance):
        if value > 0:
            count += 1
            key = features[index]
            importance_dict[key] = value
            logger.info('%d %s %.5f' % (count, key, value))
    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)