class YanivCard(object):
    """
    Card stores the suit and rank of a single card

    Note:
        The suit variable in a standard card game should be one of [S, H, D, C, BJ, RJ] meaning [Spades, Hearts, Diamonds, Clubs, Black Joker, Red Joker]
        Similarly the rank variable should be one of [A, 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K]
    """

    suit = None
    rank = None
    suits = ["C", "D", "H", "S"]
    ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    rankScoreMap = {
        "A": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 10,
        "Q": 10,
        "K": 10,
    }

    @classmethod
    def rank2int(cls, rank: str) -> int:
        if rank not in cls.ranks:
            raise Exception(f"rank `{rank}` not in ranks")

        return cls.ranks.index(rank) + 1

    @classmethod
    def rank2score(cls, rank: str) -> int:
        if rank not in cls.rankScoreMap.keys():
            raise Exception(f"rank `{rank}` not in ranks")

        return cls.rankScoreMap[rank]

    def get_card_id(self) -> int:
        rank_id = self.get_rank_id()
        suit_id = self.get_suit_id()
        return rank_id + 13 * suit_id

    def get_rank_as_int(self) -> int:
        return self.rank2int(self.rank)

    def get_score(self) -> int:
        return self.rank2score(self.rank)

    def get_suit_id(self) -> int:
        return self.suits.index(self.suit)

    def get_rank_id(self) -> int:
        return self.ranks.index(self.rank)

    def __init__(self, suit, rank):
        """Initialize the suit and rank of a card

        Args:
            suit: string, suit of the card, should be one of suits
            rank: string, rank of the card, should be one of ranks
        """
        if suit not in self.suits:
            raise ValueError("suit: {} not in suits".format(suit))
        if rank not in self.ranks:
            raise ValueError("rank: {} not in suits".format(rank))
        self.suit = suit
        self.rank = rank

    def __eq__(self, other):
        if isinstance(other, YanivCard):
            return self.rank == other.rank and self.suit == other.suit
        else:
            # don't attempt to compare against unrelated types
            return NotImplemented

    def __hash__(self):
        suit_index = YanivCard.suits.index(self.suit)
        rank_index = YanivCard.ranks.index(self.rank)
        return rank_index + 100 * suit_index

    def __lt__(self, other):
        return self.get_card_id() < other.get_card_id()

    def __str__(self):
        return self.suit + self.rank

    def __repr__(self):
        return self.__str__()
    
    def to_unicode(self):
        utf_ace_suits = [
            0x1F0D1,
            0x1F0C1,
            0x1F0B1,
            0x1F0A1,
        ]
        
        rank_id = self.get_rank_id()
        suit_id = self.get_suit_id()

        if rank_id == 0:
            ucode = utf_ace_suits[suit_id]
        else:
            ucode = utf_ace_suits[suit_id] + rank_id
        
        return chr(ucode)
