import logging
from time import time

import matplotlib.pyplot as plt

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

    if True:
        size = 0.2
        for location in range(21, 81):
            x = float(location) / 100.0
            fig, ax = plt.subplots(figsize=(0.4, 0.4))
            ax.add_artist(plt.Circle((x, x), size, color='black', fill=False))
            plt.axis('off')
            output_file = '../output/circle_frames/plotcircle{}.png'.format(location)
            logger.info('writing to %s' % output_file)
            fig.savefig(output_file)
            del fig

    else:
        location_x = 0.2
        location_y = 0.2
        size = 0.1
        random_state = 1
        for step in range(100):
            case = randint(0, 3)
            if case == 0:
                location_x += 0.01
            elif case == 1:
                location_x -= 0.01
            elif case == 2:
                location_y += 0.01
            else:
                location_y -= 0.01
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.add_artist(plt.Circle((location_x, location_y), size, color='black', fill=False))
            plt.axis('off')
            output_file = '../output/circle_frames/plotcircle{}.png'.format(step)
            logger.info('writing to %s' % output_file)
            fig.savefig(output_file)
            del fig

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
