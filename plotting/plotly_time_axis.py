import logging
from time import time

import pandas as pd
from plotly.graph_objs import Scatter
from plotly.offline import plot

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

    # load the source data from a known file
    df = pd.read_csv('../output/plotly_test_data.csv')
    df.set_index(['dates'], inplace=True)

    data = [Scatter(x=df.index, y=df.x)]
    filename = '../output/plotly_time_axis.html'
    logger.info('writing result to %s' % filename)
    plot(auto_open=False, figure_or_data=data, filename=filename, show_link=False)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
