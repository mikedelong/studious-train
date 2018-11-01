import logging
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
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
    slice_size = 200
    sqrt_periods = int(np.sqrt(float(periods)))
    start = datetime(2018, 4, 15, 0, 0, 0)
    dates = pd.date_range(start=start, periods=periods, freq='S')
    x = np.linspace(start=0, stop=periods + 1, num=periods).transpose()
    y = np.linspace(start=0, stop=sqrt_periods, num=periods).transpose()
    y = y * y
    speed = np.linspace(start=0, stop=periods + 1, num=periods).transpose()
    phenomenon = [1.0 + 0.05 * np.random.uniform(0, 1) for j in range(periods)]
    df = pd.DataFrame.from_dict(
        {'dates': dates, 'x': x, 'y': y, 'speed': speed, 'phenomenon': phenomenon}).set_index('dates')
    df['color'] = (256.0 * df['speed'] / float(periods)).astype('int32')

    if False:
        data = [dict(x=df['x'].values, y=df['phenomenon'].values, mode='markers')]
        layout = Layout(height=800, width=800, xaxis=dict(autorange=False, range=[0, periods], zeroline=True),
                        updatemenus=[{'type': 'buttons',
                                      'buttons': [{'label': 'Play',
                                                   'method': 'update',
                                                   'args': [None]}]}])
        frames = [dict(data=[
            Scatter(x=df['x'].iloc[:k], y=df['phenomenon'].iloc[:k], mode='markers',
                    marker=dict(color=df['color'].iloc[:k], colorscale='Jet', size=6))]) for k in
            range(0, periods, slice_size)]
        figure = Figure(data=data, layout=layout, frames=frames)
    else:
        traces = list()
        steps = [dict(method='restyle',
                      args=['visible', [j == k for j in range(0, periods, slice_size)]
                            ]) for k in range(0, periods, slice_size)]
        for k in range(0, periods, slice_size):
            traces.append(
                dict(type='scatter',
                     name=str(k),
                     x=df['x'].iloc[k:k + slice_size].values,
                     y=df['phenomenon'].iloc[k:k + slice_size].values,
                     mode='markers')
            )

        sliders = dict(steps=steps)
        layout = dict(sliders=[sliders], xaxis=dict(range=[0, periods]), yaxis=dict(range=[1, 1.05]))
        # data = Data(traces)
        figure = Figure(data=traces, layout=layout)
    plot(figure, filename='../output/plotly_scatter_slider.html', auto_open=False)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
