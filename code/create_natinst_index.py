# https://qbox.io/blog/building-an-elasticsearch-index-with-python
import logging
from time import time

from elasticsearch import Elasticsearch

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
    ES_HOST = {'host': 'localhost', 'port': 9200}
    # todo move index name to a setting
    INDEX_NAME = 'natinst'
    es = Elasticsearch(hosts=[ES_HOST])
    if es.indices.exists(INDEX_NAME):
        logger.info('deleting \'{}\' index'.format(INDEX_NAME))
        res = es.indices.delete(index=INDEX_NAME)
        logger.info('response: \'{}\''.format(res))
    # since we are running locally, use one shard and no replicas
    request_body = dict(settings={
        'number_of_shards': 1,
        'number_of_replicas': 0
    })
    logger.info('creating \'{}\' index'.format(INDEX_NAME))
    res = es.indices.create(index=INDEX_NAME, body=request_body)
    logger.info('response: \'{}\''.format(res))

    logger.info('done')

    finish_time = time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
