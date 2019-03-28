import abc
import logging
import os
import sys
import requests


class Parser(abc.ABC):
    def __init__(self, chandler_level=logging.INFO):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(chandler_level)

        file_handler = logging.FileHandler(filename=os.path.relpath('./logs/bash_parser.log'), mode='w')
        file_handler.setLevel(level=logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)


        self.logger.addHandler(stream_handler)
        self.logger.addHandler(file_handler)


    def load_page(self, page_url):
        return requests.get(page_url)

    @abc.abstractmethod
    def parse_page(self, page_url):
        pass

    def parse_pages(self, pages_url, **params):
        jokes = []
        for page_url in pages_url:
            self.logger.info('parse {page}'.format(page=page_url))
            try:
                jokes += self.parse_page(page_url, **params)
            except Exception as e:
                self.logger.log(logging.ERROR, e)
                self.logger.log(logging.ERROR, 'failed to parse {page}'.format(page=page_url))
                continue

        return jokes
