import logging
from glob import glob
from ntpath import basename
from time import time

from pyap import parse

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
    input_folder = '../data/text/'

    stopwords = {'Ave.', 'Camino', 'Cir.', 'Paseo', 'Pl.', 'Rd.', 'St.', 'Terr.', 'P.O.'}

    for full_input_file in glob(input_folder + '*.txt'):
        short_name = basename(full_input_file).replace('.txt', '')

        with open(full_input_file, 'rb') as input_fp:
            data = input_fp.read().decode('utf-8')
            tokens = data.split()

        found = set()
        logger.debug('our file contains %d tokens' % len(tokens))
        for start_index in range(len(tokens)):
            sample = None
            for length in range(3, 10):
                end_sample = min(start_index + length, len(tokens))
                sample = ' '.join(tokens[start_index: end_sample])
                # we probably don't want to do this; what do we want to do here?
                logger.debug(sample)
                parsed = parse(sample, country='US')
                if len(parsed) > 0:
                    full_address = parsed[0].as_dict()['full_address']
                    if full_address not in found:
                        found.add(full_address)
                        logger.info('%s start %d size %d found %s ' % (short_name, start_index, length,
                                                                       parsed[0].as_dict()))
                # does the window contain any of our tokens of interest?
                if len(stopwords.intersection(tokens[start_index: end_sample])) > 0:
                    logger.info(sample)
                if start_index % 1000 == 0 and start_index > 0:
                    logger.debug('%d %s' % (start_index, sample))

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
