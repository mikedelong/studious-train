# https://community.plot.ly/t/multiple-plots-running-on-frames/8235/8

import logging
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from plotly.graph_objs import Scatter
from plotly.offline import plot
from plotly.tools import make_subplots

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

    # first create our bogus source data
    periods = 2000
    slice_count = 10
    slice_size = periods // slice_count
    sqrt_periods = int(np.sqrt(float(periods)))
    start = datetime(2018, 4, 15, 0, 0, 0)
    dates = pd.date_range(start=start, periods=periods, freq='S')
    x = np.linspace(start=0, stop=periods + 1, num=periods).transpose()
    y = np.linspace(start=0, stop=sqrt_periods, num=periods).transpose()
    y = y * y
    z = np.linspace(start=0, stop=periods + 1, num=periods).transpose()

    speed = np.linspace(start=0, stop=periods + 1, num=periods).transpose()
    phenomenon = [1.0 + 0.05 * np.random.uniform(0, 1) for j in range(periods)]
    df = pd.DataFrame.from_dict(
        {'dates': dates, 'x': x, 'y': y, 'z': z, 'speed': speed, 'phenomenon': phenomenon}).set_index('dates')
    df['color'] = (256.0 * df['speed'] / float(periods)).astype('int32')

    figure = make_subplots(rows=4, cols=1, shared_xaxes=True, shared_yaxes=False, start_cell='top-left',
                           print_grid=True)

    figure.append_trace(Scatter(x=df['x'].values, y=df['y'].values, name='y'), 1, 1)
    figure.append_trace(Scatter(x=df['x'].values, y=df['z'].values, name='z'), 2, 1)
    figure.append_trace(Scatter(x=df['x'].values, y=df['phenomenon'].values, name='noise'), 3, 1)
    figure.append_trace(Scatter(x=df['x'].values, y=df['color'].values, name='color'), 4, 1)

    figure['layout'].update(height=600, width=600, xaxis={'rangeslider': {'visible': True}})
    plot(figure, filename='../output/plotly_lines_shared_slider.html', auto_open=False)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
