
from typing import List
from yaniv_rl import utils
from yaniv_rl.game.card import YanivCard

class YanivPlayer(object):

    def __init__(self, player_id: int, np_random):
        ''' Initilize a player.

        Args:
            player_id (int): The id of the player
        '''
        self.np_random = np_random
        self.player_id = player_id
        self.hand = [] # type: List[YanivCard]
        self.actions = []
        self.starting_hand = ""

    def get_player_id(self):
        ''' Return the id of the player
        '''

        return self.player_id
    
    def save_starting_hand(self):
        self.starting_hand = utils.cards_to_str(self.hand)