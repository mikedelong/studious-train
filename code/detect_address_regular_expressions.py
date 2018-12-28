import logging
from glob import glob
from ntpath import basename
from re import compile
from time import time

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

    txt = '44 West 22nd Street, New York, NY 12345'
    regexp = compile('[0-9]{1,3} .+, .+, [A-Z]{2} [0-9]{5}')
    address = regexp.findall(txt)
    logger.info(address)

    input_file = '../data/text/zzz.txt'
    with open(input_file, 'rb') as input_fp:
        data = input_fp.readlines()
        for item in data:
            clean_item = item.strip().decode('utf-8')
            logger.info('%s : %s' % (clean_item, regexp.findall(clean_item)))

    input_folder = '../data/text/'
    for full_input_file in glob(input_folder + '*.txt'):
        short_name = basename(full_input_file).replace('.txt', '')
        with open(full_input_file, 'rb') as input_fp:
            data = input_fp.readlines()
            for item in data:
                clean_item = item.strip().decode('utf-8')

                found = regexp.findall(clean_item)
                if len(found) > 0:
                    logger.info('%s %s : %s' % (short_name, clean_item, found))

    city_state_0_regex = compile('.+, [A-Z]{2} [0-9]{5}')
    logger.info(city_state_0_regex.findall('Ossining, NY 10520'))

    city_state_1_regex = compile('.+, [A-Z]{2}, [0-9]{5}')
    logger.info(city_state_1_regex.findall('Ossining, NY, 10520'))

    input_folder = '../data/text/'
    for full_input_file in glob(input_folder + '*.txt'):
        short_name = basename(full_input_file).replace('.txt', '')
        with open(full_input_file, 'rb') as input_fp:
            data = input_fp.readlines()
            for item in data:
                clean_item = item.strip().decode('utf-8')
                found = city_state_0_regex.findall(clean_item)
                if len(found) > 0:
                    logger.info('%s %s : %s' % (short_name, clean_item, found))
                found = city_state_1_regex.findall(clean_item)
                if len(found) > 0:
                    logger.info('%s %s : %s' % (short_name, clean_item, found))

    # let's check the least restrictive expressions against the largest, messiest document
    with open('../../../data/911-Commission-Report/911_Full_Report_djvu.txt', 'rb') as input_fp:
        data = input_fp.readlines()
        for index, item in enumerate(data):
            clean_item = item.strip().decode('utf-8')
            found = city_state_0_regex.findall(clean_item)
            if len(found) > 0:
                logger.info('%s : %s' % (clean_item, found))
            found = city_state_1_regex.findall(clean_item)
            if len(found) > 0:
                logger.info('%s : %s' % (clean_item, found))
            if index % 10000 == 0 and index > 0:
                logger.info(index)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
