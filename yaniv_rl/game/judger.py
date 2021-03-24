from rlcard.core import Card
from typing import List


class YanivJudger(object):
    def __init__(self, np_random):
        """Initialize a yaniv judger class"""
        self.np_random = np_random