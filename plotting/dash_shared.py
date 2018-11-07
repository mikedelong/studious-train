import logging
from datetime import datetime
from time import time

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from plotly.graph_objs import Scatter
from plotly.graph_objs.layout.scene import Camera
from plotly.tools import make_subplots


def get_stacked_subplots():
    result = make_subplots(rows=4, cols=1, shared_xaxes=True, shared_yaxes=False, start_cell='top-left',
                           print_grid=True)

    result.append_trace(Scatter(name='y', x=df['x'].values, y=df['y'].values), 1, 1)
    result.append_trace(Scatter(name='z', x=df['x'].values, y=df['z'].values), 2, 1)
    result.append_trace(Scatter(name='noise', x=df['x'].values, y=df['phenomenon'].values), 3, 1)
    result.append_trace(Scatter(name='color', x=df['x'].values, y=df['color'].values), 4, 1)
    result['layout'].update(height=700, width=700, xaxis={'rangeslider': {'visible': True}})

    return result


def get_stacked_subplots_no_rangeslider(arg_min, arg_max):
    result = make_subplots(rows=4, cols=1, shared_xaxes=True, shared_yaxes=False, start_cell='top-left',
                           print_grid=True)

    result.append_trace(Scatter(name='y', x=df['x'].values[arg_min:arg_max], y=df['y'].values[arg_min:arg_max]), 1, 1)
    result.append_trace(Scatter(name='z', x=df['x'].values[arg_min:arg_max], y=df['z'].values[arg_min:arg_max]), 2, 1)
    result.append_trace(
        Scatter(name='noise', x=df['x'].values[arg_min:arg_max], y=df['phenomenon'].values[arg_min:arg_max]), 3, 1)
    result.append_trace(Scatter(name='color', x=df['x'].values[arg_min:arg_max], y=df['color'].values[arg_min:arg_max]),
                        4, 1)
    result['layout'].update(height=700, legend=dict(orientation='h'), width=700)
    return result


def get_scatter3d(arg_min, arg_max):
    result = make_subplots(rows=1, cols=1, specs=[[{'is_3d': True}]])
    result.append_trace(
        dict(
            marker=dict(
                line=dict(color=df['color'].values, colorscale=colorscale, width=3),
                opacity=0.1,
                size=4,
                symbol='circle'
            ),
            type='scatter3d',
            x=df['x'].values[arg_min: arg_max],
            y=df['y'].values[arg_min: arg_max],
            z=df['z'].values[arg_min: arg_max],
            scene='scene1'
        ), 1, 1)
    result['layout'].update(height=700, width=700)
    result['layout']['scene'].update(camera=Camera(up=dict(x=0, y=0, z=1), eye=dict(x=2, y=2, z=1)))  # todo revisit
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

    # first create our bogus source data
    periods = 2000
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

    app = dash.Dash(__name__)

    colorscale = 'Jet'
    name_2d = 'datadata'
    scatter2d_marker_line = dict(color=df['color'].values, colorscale=colorscale, width=1)
    scatter2d_marker = dict(size=1, symbol='circle', line=scatter2d_marker_line, opacity=0.1)

    app.layout = html.Div([
        html.Div([
            html.Div([
                dcc.Graph(id='output-container-range-slider-2')
            ], className='five columns'),
            html.Div([
                dcc.Graph(id='output-container-range-slider'),
            ], className='seven columns'),
        ], className='row'),
        html.Div([
            dcc.RangeSlider(
                allowCross=False,
                id='range-slider-3d',
                min=0,
                marks={item: item for item in range(0, periods, 100)},
                max=periods,
                step=1,
                value=[0, periods], className='row')
        ])

    ])
    # todo figure out how to add this from static
    app.css.append_css({
        'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
    })


    @app.callback(
        dash.dependencies.Output('output-container-range-slider', 'figure'),
        [dash.dependencies.Input('range-slider-3d', 'value')])
    def update_scatter3d(arg_slider_values):
        min_value = arg_slider_values[0]
        max_value = arg_slider_values[1]
        logger.info('resizing with values %s' % arg_slider_values)
        return get_scatter3d(arg_min=min_value, arg_max=max_value)


    @app.callback(
        dash.dependencies.Output('output-container-range-slider-2', 'figure'),
        [dash.dependencies.Input('range-slider-3d', 'value')])
    def update_scatter2d(arg_slider_values):
        min_value = arg_slider_values[0]
        max_value = arg_slider_values[1]
        logger.info('resizing with values %s' % arg_slider_values)
        return get_stacked_subplots_no_rangeslider(arg_min=min_value, arg_max=max_value)


    app.run_server(debug=True, port=8051)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
