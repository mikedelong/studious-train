import logging
from datetime import timedelta
from time import time

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from plotly.graph_objs import Scatter
from plotly.tools import make_subplots


def get_stacked_scatter2d(arg_min, arg_max, arg_logger):
    arg_logger.info('arg_min: %s' % arg_min)
    arg_logger.info('arg_max: %s' % arg_max)
    t_min = df['dates'].min() + timedelta(seconds=arg_min)
    t_max = df['dates'].min() + timedelta(seconds=arg_max)
    df_local = df[(t_min <= df.dates) & (df.dates <= t_max)]
    arg_logger.info('df_local shape is %s' % str(df_local.shape))
    result = make_subplots(print_grid=False, rows=4, shared_xaxes=True, start_cell='top-left')
    x_values = df_local.dates
    arg_logger.info(len(x_values))
    for index, name in enumerate(['y', 'z', 'noise', 'color']):
        result.append_trace(Scatter(name=name, x=x_values, y=df_local[name].values), index + 1, 1)
    result['layout'].update(height=700, legend=dict(orientation='h'), width=700)
    arg_logger.info('ready to return')
    return result


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

    # load the source data from a known file
    df = pd.read_csv('../output/plotly_test_data.csv', parse_dates=['dates'])
    periods = len(df)
    logger.info('our input data has columns with the following names: %s' % list(df))

    # start the app
    # load the CSS from the local assets folder
    app = dash.Dash(__name__, include_assets_files='bWLwgP.css')
    # tell the app to load its other style sheets etc. from local storage instead trying to get them from the Web
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True

    slider_marks = {item: str(timedelta(seconds=item)) for item in range(0, periods, 100)}
    app.layout = html.Div([
        html.Div([
            html.Div([
                dcc.Graph(id='stacked-scatter-2d')
            ], className='five columns'),
        ], className='row'),
        html.Div([
            dcc.RangeSlider(allowCross=False, className='row', id='global-range-slider',
                            min=0, marks=slider_marks, max=periods, step=1,
                            value=[0, periods]
                            )])
    ])


    @app.callback(
        dash.dependencies.Output('stacked-scatter-2d', 'figure'),
        [dash.dependencies.Input('global-range-slider', 'value')])
    def update_stacked_scatter2d(arg_slider_values):
        min_value = arg_slider_values[0]
        max_value = arg_slider_values[1]
        logger.info('resizing the stacked 2d scatterplots with values %s' % arg_slider_values)
        return get_stacked_scatter2d(arg_min=min_value, arg_max=max_value, arg_logger=logger)


    port = 8052
    app.run_server(debug=True, port=port)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
