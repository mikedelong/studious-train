import logging
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
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
    start = datetime(2018, 4, 15, 0, 0, 0)
    dates = pd.date_range(start=start, periods=periods, freq='S')
    x = np.linspace(start=0, stop=periods + 1, num=periods).transpose()
    y = np.linspace(start=0, stop=periods + 1, num=periods).transpose()
    y = y * y
    z = np.linspace(start=0, stop=periods + 1, num=periods).transpose()
    speed = np.linspace(start=0, stop=periods + 1, num=periods).transpose()
    df = pd.DataFrame.from_dict({'dates': dates, 'x': x, 'y': y, 'z': z, 'speed': speed}).set_index('dates')

    trace2 = Scatter3d(
        x=df['x'].values,
        y=df['y'].values,
        z=df['z'].values,
        mode='markers',
        marker=dict(
            color='rgb(127, 127, 127)',
            size=6,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.9
        )
    )

    data = [trace2]
    fig = Figure(data=data)

    plot(fig, filename='../output/plotly_3d_animation.html', auto_open=False)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
