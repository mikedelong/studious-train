import logging
from time import time

import pandas as pd
from plotly.graph_objs import Scatter
from plotly.offline import plot
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


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

    usecols = ['Title', 'Author']
    source_df = pd.read_csv('../data/books.csv', usecols=usecols)
    logger.info(list(source_df))
    logger.info(source_df.head())

    source_df = source_df.dropna()

    source_df['modified_author'] = source_df['Author'].apply(modify_author)
    names = source_df['modified_author'].dropna().values
    bogeys = source_df['Title'].dropna().values
    # build the bigrams list; this is our feature list
    bigrams = sorted(
        list(set([item[i:i + 2] for item in names.tolist() + bogeys.tolist() for i in range(len(item) - 1)])))
    logger.info('we found %d bigrams' % len(bigrams))
    data_dict = dict()
    for item in set(names.tolist() + bogeys.tolist()):
        logger.info('%s %s' % (item, item in names))
        data_dict[item] = [int(bigram in item) for bigram in bigrams] + [int(item in names)]

    data_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=bigrams + ['target'])
    logger.info('our data has %d rows and %d columns' % data_df.shape)

    test_size = 0.33
    split_random_state = 1
    X_train, X_test, y_train, y_test = train_test_split(data_df[bigrams], data_df['target'].values,
                                                        test_size=test_size, random_state=split_random_state)

    for index, random_state in enumerate(range(1000)):
        model = DecisionTreeClassifier(random_state=random_state)
        model.fit(X=X_train, y=y_train)
        y_predicted = model.predict(X=X_test)
        current = y_predicted if index == 0 else (float(index) * current + y_predicted) * (1.0 / float(index + 1))
        if index % 10 == 0:
            logger.info('%d names: %s \nexpected: %s \nactual: %s \ncurrent: %s' %
                        (random_state, X_test.index.values, y_test, y_predicted, current))

    current_hard = (current >= 0.5).astype('float')
    logger.info('\n%s' % confusion_matrix(y_true=y_test, y_pred=current_hard))
    plot([Scatter(x=y_test, y=current, text=X_test.index.values, mode='markers+text')], auto_open=False,
         show_link=False, filename='../output/bigrams.html')
    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
