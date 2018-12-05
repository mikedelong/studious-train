import logging
from time import time

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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

    names = source_df['Author'].dropna().values
    bogeys = source_df['Title'].dropna().values
    # build the bigraphs list; this is our feature list
    bigraphs = sorted(
        list(set([item[i:i + 2] for item in names.tolist() + bogeys.tolist() for i in range(len(item) - 1)])))
    logger.info(len(bigraphs))
    logger.info(bigraphs[:20])
    data_dict = dict()
    for item in set(names.tolist() + bogeys.tolist()):
        logger.info('%s %s' % (item, item in names))
        data_dict[item] = [int(bg in item) for bg in bigraphs] + [int(item in names)]

    data_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=bigraphs + ['target'])
    logger.info(data_df.shape)

    for index, random_state in enumerate(range(100)):
        model = DecisionTreeClassifier(random_state=random_state)
        model.fit(X=data_df[bigraphs].iloc[:-2], y=data_df['target'].values[:-2])
        y_predicted = model.predict(X=data_df[bigraphs].iloc[-2:])
        current = y_predicted if index == 0 else (float(index) * current + y_predicted) * (1.0 / float(index + 1))
        if index % 10 == 0:
            logger.info('%d names: %s expected: %s actual: %s, current: %s' %
                        (random_state, data_df.index[-2:].values, data_df['target'].values[-2:], y_predicted, current))

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
