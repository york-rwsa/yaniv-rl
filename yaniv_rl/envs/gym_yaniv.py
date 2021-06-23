from rlcard.utils.utils import print_card
from yaniv_rl import utils
import numpy as np
from yaniv_rl.game import Game, player
import gym
from gym import spaces
from gym.envs.registration import register

register(
    id="Yaniv-v1",
    entry_point="yaniv_rl.envs.gym_yaniv:YanivEnv",
)

DEFAULT_GAME_CONFIG = {
    "end_after_n_deck_replacements": 0,
    "end_after_n_steps": 100,
    "early_end_reward": -1,
    "use_scaled_negative_reward": True,
    "max_negative_reward": -1,
    "negative_score_cutoff": 50,
}


class YanivEnv(gym.Env):
    metadata = {"render.modes": ["human"], "name": "Yaniv-v0"}

    def __init__(self, single_step=True, config={}):
        super(YanivEnv).__init__()
        self.single_step = single_step
        self.num_players = 2

        self.game = Game(single_step_actions=single_step, num_players=self.num_players)

        conf = DEFAULT_GAME_CONFIG.copy()
        conf.update(config)
        self.game.configure(conf)

        self.action_space = spaces.Discrete(self.game.get_action_num())
        self.observation_space = spaces.Box(shape=(266,), low=0, high=1, dtype=int)
        
        self.timestep = 0
        self.current_player = None

    def reset(self):
        state, player_id = self.game.init_game()
        self.current_player = player_id
        self.timestep = 0

        return self._get_observations(), self._get_infos()

    def step(self, action, raw_action=False):
        if not raw_action:
            action = self._decode_action(action)

        self.timestep += 1
        _, player_id = self.game.step(action)

        done = self.game.is_over()
        if done:
            rewards = self.game.get_payoffs()
        else:
            rewards = [-0.1 for _ in range(self.num_players)]

        self.current_player = player_id
        return (
            self._get_observations(),
            rewards,
            done,
            self._get_infos(),
        )

    def _get_infos(self):
        _, action_masks = self._get_legal_actions()
        infos = []
        for i in range(self.num_players):
            info = {}
            info['legal_actions'] = action_masks[i]
            info['action_mask'] = action_masks[i]
        
        return infos

    def _get_observations(self):
        observations = []
        for i in range(self.num_players):
            obs = self._extract_state(i)
            observations.append(obs)

        return observations

    def _decode_action(self, action_id):
        if self.single_step:
            return utils.JOINED_ACTION_LIST[action_id]
        else:
            return utils.ACTION_LIST[action_id]

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        if self.game._single_step_actions:
            legal_ids = [utils.JOINED_ACTION_SPACE[action] for action in legal_actions]
        else:
            legal_ids = [utils.ACTION_SPACE[action] for action in legal_actions]

        action_masks = [np.zeros(self.action_space.n) for _ in range(self.num_players)]
        np.put(action_masks[self.current_player], ind=legal_ids, v=1)

        return legal_ids, action_masks

    def get_moves(self):
        return self._get_legal_actions()[0]

    def _extract_state(self, player_id):
        if self.game.is_over():
            return np.zeros(self.observation_space.spaces[0].shape)

        discard_pile = self.game.round.discard_pile
        if self.game.round.discarding:
            last_discard = discard_pile[-1]
        else:
            last_discard = discard_pile[-2]

        available_discard = set([last_discard[0], last_discard[-1]])
        deadcards = [c for d in discard_pile for c in d if c not in available_discard]

        current_player = self.game.players[player_id]
        next_player = self.game.players[self.game.round._get_next_player(player_id)]
        known_cards = self.game.round.known_cards[player_id]
        unknown_cards = self.game.round.dealer.deck + [
            c for c in next_player.hand if c not in known_cards
        ]

        card_obs = [
            current_player.hand,
            available_discard,
            deadcards,
            known_cards,
            unknown_cards,
        ]
        card_obs = np.ravel(list(map(utils.encode_cards, card_obs)))

        opponent_hand_size = np.zeros(6)
        opponent_hand_size[len(next_player.hand)] = 1

        obs = np.concatenate((card_obs, opponent_hand_size))

        return obs

    def render(self, mode="human"):
        state = self.game.get_state(self.current_player)
        _print_state(state, [])

    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)

def _print_state(state, action_record):
    """Print out the state of a given player

    Args:
        player (int): Player id
    """
    _action_list = []
    for i in range(1, len(action_record) + 1):
        if action_record[-i][0] == state["current_player"]:
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print(">> Player", pair[0], "chose ", end="")
        print(pair[1])

    print("")

    print("============= Discard pile ==============")
    print(", ".join(["  ".join([c for c in cards]) for cards in state["discard_pile"]]))

    if "pickup_top_discard" in state["legal_actions"]:
        availcards = "  ".join([c for c in state["discard_pile"][-2]])
        print(f"You can pick up: {availcards}")
    else:
        print()
    print()
    print("============== Your Hand ================")
    print("  ".join([c for c in state["hand"]]))
    print("")
    print("============= Opponents Hand ============")
    for i in range(state["player_num"]):
        if i != state["current_player"]:
            print(
                "Player {} has {} cards: {}".format(
                    i, state["hand_lengths"][i], " ".join(state["known_cards"][i])
                )
            )