# https://community.plot.ly/t/multiple-plots-running-on-frames/8235/8

import logging
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
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

    # first create our bogus source data
    periods = 2000
    slice_size = 1000
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

    # graph constants
    colorscale = 'Jet'
    name_2d = 'phenomenon'
    name_3d = 'location'
    # make the left-hand scatter plot data
    scatter2d_marker_line = dict(color=df['color'].values, colorscale=colorscale, width=1)
    scatter2d_marker = dict(size=1, symbol='circle', line=scatter2d_marker_line, opacity=0.9)
    scatter2d = {
        'marker': scatter2d_marker,
        'mode': 'markers',
        'name': name_2d,
        'x': df['x'].values,
        'y': df['phenomenon'].values,
        'type': 'scatter',
    }

    # then make the right-hand 3d scatter plot data
    scatter3d_marker_line = dict(color=df['color'].values, colorscale=colorscale, width=1)
    scatter3d_marker = dict(size=1, symbol='circle', line=scatter3d_marker_line, opacity=0.9)
    scatter3d = {
        'marker': scatter3d_marker,
        'mode': 'markers',
        'name': name_3d,
        'type': 'scatter3d',
        'x': df['x'].values,
        'y': df['y'].values,
        'z': df['z'].values
    }

    figure = dict(
        layout=dict(
            xaxis1={'domain': [0.0, 0.5], 'anchor': 'y1', 'title': '1', 'range': [0, periods]},
            yaxis1={'domain': [0.0, 1.0], 'anchor': 'x1', 'title': 'y', 'range': [0.99, 1.06]},
            # xaxis2={'domain': [0.5, 1.0], 'anchor': 'y2', 'title': '2', },
            # yaxis2={'domain': [0.0, 1.0], 'anchor': 'x2', 'title': 'y', },
            scene={'domain': {'x': [0.5, 1], 'y': [0, 1]},
                   'xaxis': dict(nticks=4, range=[0, periods], autorange=False),
                   'yaxis': dict(nticks=4, range=[0, periods], autorange=False),
                   'zaxis': dict(nticks=4, range=[0, periods], autorange=False)},
            title='',
            margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
            sliders=[{'yanchor': 'top',
                      'xanchor': 'left',
                      'visible': True,
                      'transition': {'duration': 100.0, 'easing': 'linear'},
                      'pad': {'b': 10, 't': 50},
                      'len': 0.9,
                      'x': 0.1, 'y': 0,
                      'steps': [{
                          'args': [[str(k)], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False},
                                              'transition': {'duration': 0, 'easing': 'linear'}}],
                          'method': 'animate',
                          'label': str(k)
                      } for k in range(0, periods, slice_size)]}]
        ),

        data=[scatter2d, scatter3d],
        frames=[
            {
                'name': str(k),
                # 'layout': dict(),
                'data': [{
                    'marker': scatter2d_marker,
                    'mode': 'markers',
                    'name': name_2d,
                    'type': 'scatter',
                    'x': df['x'].values[k:k + slice_size],
                    'xaxis': 'x1',
                    'y': df['phenomenon'].values[k:k + slice_size],
                    'yaxis': 'y1'
                }, {
                    'marker': scatter3d_marker,
                    'mode': 'lines',
                    'name': name_3d,
                    'type': 'scatter3d',
                    'x': df['x'].values[k:k + slice_size],
                    'y': df['y'].values[k:k + slice_size],
                    'z': df['z'].values[k:k + slice_size]
                }]
            }
            for k in range(0, periods, slice_size)
        ]

    )

    plot(figure, filename='../output/plotly_common_slider.html', auto_open=False)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
