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

    text = 'General O\'Brien was commissioned in 1981 through the Air Force Reserve Officer Training Corps upon ' \
           'graduation from the University of New Hampshire. He has served in a wide variety of development, ' \
           'acquisition and operations roles within the Air Force and the National Reconnaissance Office. His ' \
           'active-duty career included assignments in requirements development, engineering and program management, ' \
           'as well as serving as a principal contracting officer for computers, communications, directed energy and ' \
           'national security space systems, intelligence, surveillance, reconnaissance and policy. His Reserve ' \
           'leadership responsibilities include network communications, test, maintenance, satellite on-orbit ' \
           'operations, acquisition and space launch. Prior to his current assignment, General O\'Brien was the ' \
           'Mobilization Assistant to the Commander, Space and Missiles System Center who also serves as the Air Force ' \
           'Program Executive Officer for space systems.'
    logger.info(text)
    tagged_text = preprocess(text)
    logger.info(tagged_text)

    parts_tagged = {
        item[0]: item[1] for item in tagged_text
    }

    for key, value in parts_tagged.items():
        if value not in ['VBD', 'VBN', 'IN', 'DT', 'WP', 'VBZ', 'RB', 'JJ', 'TO', 'VBG', 'VBP', 'CC', ',', '.']:
            logger.info('token {} has tag {}'.format(key, value))

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
