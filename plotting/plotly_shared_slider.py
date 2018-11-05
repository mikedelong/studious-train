# https://community.plot.ly/t/multiple-plots-running-on-frames/8235/8
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

    figure = dict(
        layout=dict(
            xaxis1={'domain': [0.0, 0.44], 'anchor': 'y1', 'title': '1', 'range': [-2.25, 3.25]},
            yaxis1={'domain': [0.0, 1.0], 'anchor': 'x1', 'title': 'y', 'range': [-1, 11]},
            xaxis2={'domain': [0.56, 1.0], 'anchor': 'y2', 'title': '2', 'range': [-2.25, 3.25]},
            yaxis2={'domain': [0.0, 1.0], 'anchor': 'x2', 'title': 'y', 'range': [-1, 11]},
            title='',
            margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
            updatemenus=[{'buttons': [{'args': [['0', '1', '2', '3'],
                                                {'frame': {'duration': 500.0, 'redraw': False}, 'fromcurrent': True,
                                                 'transition': {'duration': 0, 'easing': 'linear'}}], 'label': 'Play',
                                       'method': 'animate'}, {'args': [[None],
                                                                       {'frame': {'duration': 0, 'redraw': False},
                                                                        'mode': 'immediate',
                                                                        'transition': {'duration': 0}}],
                                                              'label': 'Pause', 'method': 'animate'}],
                          'direction': 'left', 'pad': {'r': 10, 't': 85}, 'showactive': True, 'type': 'buttons',
                          'x': 0.1, 'y': 0, 'xanchor': 'right', 'yanchor': 'top'}],
            sliders=[{'yanchor': 'top', 'xanchor': 'left',
                      'currentvalue': {'font': {'size': 16}, 'prefix': 'Frame: ', 'visible': True, 'xanchor': 'right'},
                      'transition': {'duration': 500.0, 'easing': 'linear'}, 'pad': {'b': 10, 't': 50}, 'len': 0.9,
                      'x': 0.1, 'y': 0,
                      'steps': [{'args': [['0'], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False},
                                                  'transition': {'duration': 0, 'easing': 'linear'}}], 'label': '0',
                                 'method': 'animate'},
                                {'args': [['1'], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False},
                                                  'transition': {'duration': 0, 'easing': 'linear'}}], 'label': '1',
                                 'method': 'animate'},
                                {'args': [['2'], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False},
                                                  'transition': {'duration': 0, 'easing': 'linear'}}], 'label': '2',
                                 'method': 'animate'},
                                {'args': [['3'], {'frame': {'duration': 500.0, 'easing': 'linear', 'redraw': False},
                                                  'transition': {'duration': 0, 'easing': 'linear'}}], 'label': '3',
                                 'method': 'animate'},
                                ]}]
        ),

        data=[
            {'type': 'scatter', 'name': 'f1', 'x': [-2., -1., 0.01, 1., 2., 3.], 'y': [4, 1, 1, 1, 4, 9],
             'hoverinfo': 'name+text',
             'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
             'line': {'color': 'rgba(255,79,38,1.000000)'}, 'mode': 'markers+lines',
             'fillcolor': 'rgba(255,79,38,0.600000)', 'legendgroup': 'f1', 'showlegend': True, 'xaxis': 'x1',
             'yaxis': 'y1'},
            {'type': 'scatter', 'name': 'f2', 'x': [-2., -1., 0.01, 1., 2., 3.], 'y': [2.5, 1, 1, 1, 2.5, 1],
             'hoverinfo': 'name+text',
             'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
             'line': {'color': 'rgba(79,102,165,1.000000)'}, 'mode': 'markers+lines',
             'fillcolor': 'rgba(79,102,165,0.600000)', 'legendgroup': 'f2', 'showlegend': True, 'xaxis': 'x2',
             'yaxis': 'y2'},
        ],

        frames=[
            {'name': '0', 'layout': {},
             'data': [
                 {'type': 'scatter', 'name': 'f1', 'x': [-2., -1., 0.01, 1., 2., 3.], 'y': [5, 8, 3, 2, 4, 0],
                  'hoverinfo': 'name+text',
                  'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
                  'line': {'color': 'rgba(255,79,38,1.000000)'}, 'mode': 'markers+lines',
                  'fillcolor': 'rgba(255,79,38,0.600000)', 'legendgroup': 'f1', 'showlegend': True, 'xaxis': 'x1',
                  'yaxis': 'y1'},
                 {'type': 'scatter', 'name': 'f2', 'x': [-2., -1., 0.01, 1., 2., 3.], 'y': [3, 7, 4, 8, 5, 9],
                  'hoverinfo': 'name+text',
                  'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
                  'line': {'color': 'rgba(79,102,165,1.000000)'}, 'mode': 'markers+lines',
                  'fillcolor': 'rgba(79,102,165,0.600000)', 'legendgroup': 'f2', 'showlegend': True, 'xaxis': 'x2',
                  'yaxis': 'y2'}],
             },

            {'name': '1', 'layout': {},
             'data': [
                 {'type': 'scatter', 'name': 'f1', 'x': [-2., -1., 0.01, 1., 2., 3.], 'y': [4, 1, 1, 1, 4, 9],
                  'hoverinfo': 'name+text',
                  'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
                  'line': {'color': 'rgba(255,79,38,1.000000)'}, 'mode': 'markers+lines',
                  'fillcolor': 'rgba(255,79,38,0.600000)', 'legendgroup': 'f1', 'showlegend': True, 'xaxis': 'x1',
                  'yaxis': 'y1'},
                 {'type': 'scatter', 'name': 'f2', 'x': [-2., -1., 0.01, 1., 2., 3.], 'y': [2.5, 1, 1, 1, 2.5, 1],
                  'hoverinfo': 'name+text',
                  'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
                  'line': {'color': 'rgba(79,102,165,1.000000)'}, 'mode': 'markers+lines',
                  'fillcolor': 'rgba(79,102,165,0.600000)', 'legendgroup': 'f2', 'showlegend': True, 'xaxis': 'x2',
                  'yaxis': 'y2'}],
             },

            {'name': '2', 'layout': {},
             'data': [
                 {'type': 'scatter', 'name': 'f1', 'x': [-2., -1., 0.01, 1., 2., 3.], 'y': [5, 8, 3, 2, 4, 0],
                  'hoverinfo': 'name+text',
                  'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
                  'line': {'color': 'rgba(255,79,38,1.000000)'}, 'mode': 'markers+lines',
                  'fillcolor': 'rgba(255,79,38,0.600000)', 'legendgroup': 'f1', 'showlegend': True, 'xaxis': 'x1',
                  'yaxis': 'y1'},
                 {'type': 'scatter', 'name': 'f2', 'x': [-2., -1., 0.01, 1., 2., 3.], 'y': [3, 7, 4, 8, 5, 9],
                  'hoverinfo': 'name+text',
                  'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
                  'line': {'color': 'rgba(79,102,165,1.000000)'}, 'mode': 'markers+lines',
                  'fillcolor': 'rgba(79,102,165,0.600000)', 'legendgroup': 'f2', 'showlegend': True, 'xaxis': 'x2',
                  'yaxis': 'y2'}],
             },

            {'name': '3', 'layout': {},
             'data': [
                 {'type': 'scatter', 'name': 'f1', 'x': [-2., -1., 0.01, 1., 2., 3.], 'y': [4, 1, 1, 1, 4, 9],
                  'hoverinfo': 'name+text',
                  'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
                  'line': {'color': 'rgba(255,79,38,1.000000)'}, 'mode': 'markers+lines',
                  'fillcolor': 'rgba(255,79,38,0.600000)', 'legendgroup': 'f1', 'showlegend': True, 'xaxis': 'x1',
                  'yaxis': 'y1'},
                 {'type': 'scatter', 'name': 'f2', 'x': [-2., -1., 0.01, 1., 2., 3.], 'y': [2.5, 1, 1, 1, 2.5, 1],
                  'hoverinfo': 'name+text',
                  'marker': {'opacity': 1.0, 'symbol': 'circle', 'line': {'width': 0, 'color': 'rgba(50,50,50,0.8)'}},
                  'line': {'color': 'rgba(79,102,165,1.000000)'}, 'mode': 'markers+lines',
                  'fillcolor': 'rgba(79,102,165,0.600000)', 'legendgroup': 'f2', 'showlegend': True, 'xaxis': 'x2',
                  'yaxis': 'y2'}],
             }
        ]
    )
    plot(figure, filename='../output/plotly_shared_slider.html', auto_open=False)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
