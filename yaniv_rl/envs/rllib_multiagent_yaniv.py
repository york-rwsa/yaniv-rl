from yaniv_rl import utils
import numpy as np
import copy
from collections import Counter
from ray import rllib
from gym.spaces import Box, Dict, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from yaniv_rl.game import Game

DEFAULT_GAME_CONFIG = {
    "end_after_n_deck_replacements": 0,
    "end_after_n_steps": 100,
    "early_end_reward": -1,
    "use_scaled_negative_reward": True,
    "max_negative_reward": -1,
    "negative_score_cutoff": 50,
}


class YanivEnv(MultiAgentEnv):
    def __init__(self, config={}, single_step=True):
        super(YanivEnv).__init__()
        self.single_step = single_step
        self.num_players = 2

        self.game = Game(single_step_actions=single_step, num_players=self.num_players)

        conf = DEFAULT_GAME_CONFIG.copy()
        conf.update(config)
        self.game.configure(conf)

        self.action_space = Discrete(self.game.get_action_num())
        self.observation_space = Dict(
            {
                "action_mask": Box(0, 1, shape=(self.action_space.n,)),
                "state": Box(shape=(266,), low=0, high=1, dtype=int),
            }
        )
        self.reward_range = (-1.0, 1.0)

        self.timestep = 0
        self.current_player = None

    def reset(self):
        _, player_id = self.game.init_game()
        self.current_player = player_id
        self.timestep = 0

        return self._get_observations()

    def step(self, action_dict):
        action = action_dict[self._get_player_string(self.current_player)]
        action = self._decode_action(action)

        self.game.step(action)

        done = self.game.is_over()
        dones = {p: done for p in self._get_players()}
        dones["__all__"] = done
        if done:
            payoffs = self.game.get_payoffs()
            rewards = {
                self._get_player_string(i): payoffs[i] for i in range(self.num_players)
            }
        else:
            rewards = {p: -0.1 for p in self._get_players()}

        infos = {p: {} for p in self._get_players()}

        self.timestep += 1
        self.current_player = self.game.round.current_player

        return (
            self._get_observations(),
            rewards,
            dones,
            infos,
        )

    def _decode_action(self, action_id):
        if self.single_step:
            return utils.JOINED_ACTION_LIST[action_id]
        else:
            return utils.ACTION_LIST[action_id]

    def _get_observations(self):
        observations = {}
        for i in range(self.num_players):
            obs = {
                "state": self._extract_state(i),
                "action_mask": self._get_action_mask(i),
            }
            observations[self._get_player_string(i)] = obs

        return observations

    def _get_action_mask(self, player_id):
        if player_id != self.current_player:
            return np.zeros(self.action_space.n)

        legal_actions = self.game.get_legal_actions()
        if self.game._single_step_actions:
            legal_ids = [utils.JOINED_ACTION_SPACE[action] for action in legal_actions]
        else:
            legal_ids = [utils.ACTION_SPACE[action] for action in legal_actions]

        action_mask = np.zeros(self.action_space.n)
        np.put(action_mask, ind=legal_ids, v=1)

        return action_mask

    def _extract_state(self, player_id):
        if self.game.is_over():
            return np.zeros((266,))

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

    def _get_player_string(self, id):
        return "player_{}".format(id)

    def _get_players(self):
        return [self._get_player_string(i) for i in range(self.num_players)]