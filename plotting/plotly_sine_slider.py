import logging
from time import time

import numpy as np
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

    active = 1
    data = [dict(
        visible=False,
        line=dict(
            # color='#00CED1',
            width=6),
        name='v = ' + str(step),
        x=np.arange(0, 10, 0.01),
        y=np.sin(step * np.arange(0, 10, 0.01))) for step in np.arange(0, 5, 0.1)]
    data[active]['visible'] = True

    steps = []
    for i in range(len(data)):
        step = dict(
            method='restyle',
            args=['visible', [False] * len(data)],
        )
        step['args'][1][i] = True  # Toggle i'th trace to 'visible'
        steps.append(step)

    sliders = [dict(
        active=active,
        currentvalue={'prefix': 'Frequency: '},
        pad={'t': 50},
        steps=steps
    )]

    layout = dict(sliders=sliders)

    fig = dict(data=data, layout=layout)

    plot(figure_or_data=fig, filename='sinewave_with_slider.html')
    
    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
