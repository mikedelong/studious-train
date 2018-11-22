import logging
from time import time

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

    data = [{
        'marker': dict(
            color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',
                   'rgb(44, 160, 101)', 'rgb(255, 65, 54)'],
            opacity=[0.9, 0.2, 0.9, 0.2],
            size=[40, 60, 80, 100],
        ),
        'mode': 'markers',
        'type': 'scatter',
        'x': [1, 2, 3, 4],
        'y': [10, 11, 12, 13]
    }]
    plot(data, filename='../output/plotly_variable_opacity.html', auto_open=False)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
