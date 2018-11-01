import logging
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
from plotly.graph_objs import Layout
from plotly.graph_objs import Scatter
from plotly.graph_objs import Scatter3d
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

    periods = 2000
    slice_size = 50
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

    scatter = Scatter(x=df['x'].values, y=df['phenomenon'].values, mode='markers', marker=dict(size=2, symbol='circle'))
    scatter2 = Scatter(x=df['x'].values, y=df['phenomenon'].values, mode='markers',
                       marker=dict(size=2, symbol='circle'),
                       xaxis='x2', yaxis='y2')

    scatter3d_line = dict(color=df['color'].values, colorscale='Jet', width=1)
    scatter3d = Scatter3d(x=df['x'].values, y=df['y'].values, z=df['z'].values, mode='markers',
                          marker=dict(size=2, symbol='circle', line=scatter3d_line, opacity=0.9))

    layout = Layout(height=700, width=1000, title='...', showlegend=True,
                    scene={'domain': {'x': [0.5, 1], 'y': [0, 1]}},
                    xaxis={'anchor': 'x', 'domain': [0, 0.5]},
                    yaxis={'anchor': 'y', 'domain': [0, 0.5]},
                    xaxis2={'anchor': 'x', 'domain': [0, 0.5]},
                    yaxis2={'anchor': 'y', 'domain': [0.5, 1]},
                    margin={'r': 50, 't': 50, 'b': 50, 'l': 50})

    figure = Figure(data=[scatter, scatter2, scatter3d], layout=layout)
    plot(figure, filename='../output/plotly_scatter_scatter3d.html', auto_open=False)
    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
