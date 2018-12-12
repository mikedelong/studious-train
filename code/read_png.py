import logging
from glob import glob
from time import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from plotly.graph_objs import Figure
from plotly.graph_objs import Scatter
from plotly.offline import plot


def mse(left, right):
    # is this really the best way to do this?
    err = np.sum((left.astype('float') - right.astype('float')) ** 2)
    err /= float(left.shape[0] * left.shape[1])
    return err


def get_ply(arg_list, arg_index):
    return [value[..., arg_index] for value in arg_list]


def get_mean(arg_list):
    if arg_list is None:
        return None
    if len(arg_list) == 0:
        return None
    weight = 1.0 / float(len(arg_list))
    result = np.zeros(arg_list[0].shape, np.float)
    for item in arg_list:
        result += weight * item
    return result


if __name__ == '__main__':
    start_time = time()

    console_formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    logger.info('started')
    input_folder = '../../../data/MoonPix/'

    limit = 900
    images = [np.array(Image.open(item), dtype=np.float)
              for index, item in enumerate(glob(input_folder + '*.jpg')) if index < limit]

    width, height, depth = images[0].shape

    logger.info('we read %d images and each one is %d x %d x 3' % (len(images), width, height))

    images_blue = get_ply(images, 0)
    mean_blue = get_mean(images_blue)
    errors_blue = [mse(images_blue[index], mean_blue) for index in range(len(images_blue))]
    images_green = get_ply(images, 1)
    mean_green = get_mean(images_green)
    errors_green = [mse(images_green[index], mean_green) for index in range(len(images_green))]
    images_red = get_ply(images, 2)
    mean_red = get_mean(images_red)
    errors_red = [mse(images_red[index], mean_red) for index in range(len(images_red))]

    do_3d_plot = False
    if do_3d_plot:
        figure = plt.figure()
        axis = Axes3D(figure)
        axis.scatter(xs=errors_blue, ys=errors_green, zs=errors_red)
        plt.show()
    else:
        figure = Figure(data=[
            Scatter(x=list(range(len(errors_blue))), y=errors_blue, mode='markers', marker=dict(size=3, color='blue'),
                    name='blue'),
            Scatter(x=list(range(len(errors_green))), y=errors_green, mode='markers',
                    marker=dict(size=3, color='green'), name='green'),
            Scatter(x=list(range(len(errors_red))), y=errors_red, mode='markers', marker=dict(size=3, color='red'),
                    name='red'),
        ])
        plot(figure, filename='../output/read_png.html', auto_open=False, show_link=False)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
