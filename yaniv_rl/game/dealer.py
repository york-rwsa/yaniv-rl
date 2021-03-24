
from yaniv_rl.utils import init_deck
from yaniv_rl.game.card import YanivCard
from typing import List

class YanivDealer(object):
    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = init_deck() # type: List[YanivCard]
        self.shuffle()

    def shuffle(self):
        ''' Shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def draw_card(self) -> YanivCard:
        ''' Deal one card from the deck

        Returns:
            (Card): The drawn card from the deck
        '''
        return self.deck.pop()

