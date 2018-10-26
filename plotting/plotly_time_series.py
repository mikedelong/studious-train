import logging
from time import time

import pandas as pd
import plotly
import plotly.graph_objs as go

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

    data_folder = '../data/'
    output_folder = '../output/'

    aapl_df = pd.read_csv('../data/bokeh/AAPL.csv')
    logger.info(aapl_df.shape)
    logger.info(list(aapl_df))

    aapl_time_series = go.Scatter(x=aapl_df['Date'], y=aapl_df['Close'], name='Close', opacity=0.8)
    # aapl_time_series = [
    #     go.Scatter(x=aapl_df['Date'], y=aapl_df[item], name=item, opacity=0.8)
    #     for item in list(aapl_df) if item not in ['Date', 'Volume']
    # ]

    aapl_scatter = go.Scatter(x=aapl_df['Open'].values, y=aapl_df['Close'].values, mode='markers',
                              name='DailyChange', hoverinfo='name',
                              marker=dict(color=aapl_df['Volume'].values, size=2, showscale=False,
                                          colorscale='Viridis'))

    fig = plotly.tools.make_subplots(rows=2, cols=1)
    fig.append_trace(aapl_time_series, 1, 1)
    fig.append_trace(aapl_scatter, 2, 1)

    plotly.offline.plot(fig, filename=output_folder + 'plotly_time_series.html')
    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
