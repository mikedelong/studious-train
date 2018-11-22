import logging
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
from plotly.graph_objs import Figure
from plotly.graph_objs import Layout
from plotly.graph_objs import Surface
from plotly.graph_objs.layout.scene import Camera


from plotly.graph_objs.layout import Scene
from plotly.offline import plot


def update_z(frequency):
    f.data[0].z = np.cos(x * yt * frequency / 10.0) + np.sin(x * yt * frequency / 10.0) * 2


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

    x = y = np.arange(-5, 5, 0.1)
    yt = x[:, np.newaxis]
    z = np.cos(x * yt) + np.sin(x * yt) * 2

    f = Figure(
        data=[
            Surface(z=z, x=x, y=y,colorscale='Viridis')],
        layout=Layout(scene=Scene(
            camera=Camera(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.25, y=1.25, z=1.25))
        ))
    )

    plot(f, filename='out.html')


    # freq_slider = interactive(update_z, frequency=(1, 50, 0.1))
    # vb = VBox((f, freq_slider))
    # vb.layout.align_items = 'center'

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
