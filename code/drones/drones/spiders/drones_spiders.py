import scrapy
import hashlib

hasher = hashlib.md5()
limit = 10
urls = [
    'https://en.wikipedia.org/wiki/Unmanned_combat_aerial_vehicle',
    # 'https://nationalinterest.org/tag/drones',
    # 'https://www.dedrone.com/blog',
    # 'https://www.dronefly.com/blogs/news/',
    # 'https://blog.feedspot.com/drone_blogs/'
]


class MySpider(scrapy.Spider):
    name = 'drones'

    def start_requests(self):
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    custom_settings = {
        'DEPTH_LIMIT': 1
    }

    def parse(self, response):
        for next_page in response.css('div.mw-parser-output > p > a'):
            yield response.follow(next_page, self.parse)

        for quote in response.css('div.mw-parser-output > p'):
            yield {'quote': quote.extract()}
