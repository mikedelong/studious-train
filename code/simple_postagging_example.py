# https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy
import logging
from time import time

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


def preprocess(arg_sent):
    sent = word_tokenize(arg_sent)
    return pos_tag(sent)


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

    text = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile ' \
           'phone market and ordered the company to alter its practices.'

    logger.info(text)
    tagged_text = preprocess(text)
    logger.info(tagged_text)

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
