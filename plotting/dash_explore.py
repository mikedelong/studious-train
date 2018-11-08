import base64
import io
import logging
from time import time

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
from plotly.graph_objs import Scatter
from plotly.graph_objs.layout.scene import Camera
from plotly.tools import make_subplots


def get_stacked_scatter2d(arg_min, arg_max):
    if len(df) > 0:
        result = make_subplots(print_grid=False, rows=4, shared_xaxes=True, start_cell='top-left')

        x_values = df['x'].values[arg_min:arg_max]
        for index, name in enumerate(['y', 'z', 'noise', 'color']):
            result.append_trace(Scatter(name=name, x=x_values, y=df[name].values[arg_min:arg_max]), index + 1, 1)
        result['layout'].update(height=700, legend=dict(orientation='h'), width=700)
    else:
        result = make_subplots(print_grid=False, rows=4, shared_xaxes=True, start_cell='top-left')
        result['layout'].update(height=700, legend=dict(orientation='h'), width=700)
    return result


def get_scatter3d(arg_colorscale, arg_min, arg_max):
    if len(df) > 0:
        result = make_subplots(print_grid=False, specs=[[{'is_3d': True}]])
        result.append_trace(
            dict(
                marker=dict(
                    line=dict(color=df['color'].values, colorscale=arg_colorscale, width=3),
                    opacity=0.1,
                    size=4,
                    symbol='circle'
                ),
                type='scatter3d',
                x=df['x'].values[arg_min: arg_max],
                y=df['y'].values[arg_min: arg_max],
                z=df['z'].values[arg_min: arg_max]
            ), 1, 1)
        result['layout'].update(height=700, width=700)
        result['layout']['scene'].update(camera=Camera(up=dict(x=0, y=0, z=1), eye=dict(x=2, y=2, z=1)))  # todo revisit
    else:
        result = make_subplots(print_grid=False, specs=[[{'is_3d': True}]])
        result['layout'].update(height=700, width=700)
    return result


def parse_contents(contents, filename, date=None):
    logger.info('contents: %s' % contents)
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    df.set_index(['dates'], inplace=True)
    periods = len(df)

    return html.Div([
        html.H5(filename),
    ])


df = pd.DataFrame()

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

    # start the app
    # load the CSS from the local assets folder
    app = dash.Dash(__name__, include_assets_files='bWLwgP.css')
    # tell the app to load its other style sheets etc. from local storage instead trying to get them from the Web
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True
    periods = 100
    app.layout = html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
        html.Div(id='output-data-upload'),
        html.Div([
            html.Div([
                dcc.Graph(id='stacked-scatter-2d')
            ], className='five columns'),
            html.Div([
                dcc.Graph(id='scatter-3d'),
            ], className='seven columns'),
        ], className='row'),
        html.Div([
            dcc.RangeSlider(allowCross=False, className='row', id='global-range-slider', min=0,
                            marks={item: item for item in range(0, periods, 100)},
                            max=periods, step=1, value=[0, periods])])
    ])


    @app.callback(
        dash.dependencies.Output('scatter-3d', 'figure'),
        [dash.dependencies.Input('global-range-slider', 'value')])
    def update_scatter3d(arg_slider_values):
        min_value = arg_slider_values[0]
        max_value = arg_slider_values[1]
        logger.info('resizing the 3d scatterplot with values %s' % arg_slider_values)
        colorscale = 'Jet'
        return get_scatter3d(arg_colorscale=colorscale, arg_min=min_value, arg_max=max_value)


    @app.callback(
        dash.dependencies.Output('stacked-scatter-2d', 'figure'),
        [dash.dependencies.Input('global-range-slider', 'value')])
    def update_stacked_scatter2d(arg_slider_values):
        min_value = arg_slider_values[0]
        max_value = arg_slider_values[1]
        logger.info('resizing the stacked 2d scatterplots with values %s' % arg_slider_values)
        return get_stacked_scatter2d(arg_min=min_value, arg_max=max_value)


    @app.callback(Output('output-data-upload', 'children'),
                  [Input('upload-data', 'contents')],
                  [State('upload-data', 'filename'),
                   State('upload-data', 'last_modified')])
    def update_output(list_of_contents, list_of_names, list_of_dates):
        logger.info('list of contents: %s' % list_of_contents)
        logger.info('list of names: %s' % list_of_names)
        logger.info('list of dates: %s' % list_of_dates)
        if list_of_dates is None:
            list_of_dates = ['foo']
        if list_of_contents is not None:
            children = [
                parse_contents(c, n, d) for c, n, d in
                zip(list_of_contents, list_of_names, list_of_dates)]
            return children


    port = 8053
    app.run_server(debug=True, port=port)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
