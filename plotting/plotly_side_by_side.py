import logging
from time import time

from plotly.graph_objs import Figure
from plotly.graph_objs import Layout
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

    data = [
        Scatter(
            x=[1, 2, 3],
            y=[4, 5, 6]
        ),
        Scatter(
            x=[20, 30, 40],
            y=[50, 60, 70],
            xaxis='x2',
            yaxis='y2'
        )
    ]

    layout = Layout(
        xaxis=dict(
            domain=[0, 0.7]
        ),
        xaxis2=dict(
            domain=[0.8, 1]
        ),
        yaxis2=dict(
            anchor='x2'
        )
    )
    figure = Figure(data=data, layout=layout)
    plot(figure, filename='../output/plotly_side_by_side.html', auto_open=False)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
