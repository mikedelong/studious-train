import logging
from time import time

from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

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

    return_X_y = True
    (data, target) = load_diabetes(return_X_y=return_X_y)
    logger.info('data has %d rows and %d columns' % data.shape)

    model = DecisionTreeRegressor()

    test_size = 0.10
    for test_size in [0.7, 0.8, 0.9]:
        accuracies = list()
        scores = list()
        for random_state in range(500):
            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size,
                                                                random_state=random_state)
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            y_predicted = model.predict(X=X_test)
            accuracy = accuracy_score(y_test, y_predicted)
            accuracies.append(accuracy)
            scores.append(score)
        logger.info('with test size %.2f we have score %.3f and accuracy %.3f' %
                    (test_size, sum(scores) / len(scores), sum(accuracies) / len(accuracies)))

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
