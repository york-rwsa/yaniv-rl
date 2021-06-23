import numpy as np

from rlcard.envs import Env
from yaniv_rl.game import Game
from yaniv_rl import utils

DEFAULT_GAME_CONFIG = {
    "n_players": 2,
    "state_n_players": 2,
    "end_after_n_deck_replacements": 0,
    "end_after_n_steps": 100,
    "early_end_reward": 0,
    "use_scaled_negative_reward": False,
    "use_scaled_positive_reward": False,
    "max_negative_reward": -1,
    "negative_score_cutoff": 50,
    "single_step": True,
    "step_reward": 0,
    "use_unkown_cards_in_state": True,
    "use_dead_cards_in_state": True,
    "observation_scheme": 0,
    "starting_player": "random",
    "starting_hands": {}
}

def calculate_reward(state, next_state, action):
    if action not in utils.pickup_actions and len(action) > 2:
        return 0.1 * len(action) / 2
    else:
        return -0.05


# for two player
class YanivEnv(Env):
    def __init__(self, config={}):
        self.name = "yaniv"
        self.single_step = config.get("single_step_actions", False)
        self.game = Game(single_step_actions=self.single_step)
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.reward_func = calculate_reward
        # configure game
        super().__init__(config)
        self.state_shape = [266]

        _game_config = self.default_game_config.copy()
        for key in config:
            if key in _game_config:
                _game_config[key] = config[key]
        self.game.configure(_game_config)

    def _extract_state(self, state):
        if self.game.is_over():
            return {
                "obs": np.zeros(self.state_shape),
                "legal_actions": self._get_legal_actions(),
            }

        discard_pile = self.game.round.discard_pile
        if self.game.round.discarding:
            last_discard = discard_pile[-1]
        else:
            last_discard = discard_pile[-2]

        available_discard = set([last_discard[0], last_discard[-1]])
        deadcards = [c for d in discard_pile for c in d if c not in available_discard]

        current_player = self.game.players[self.game.round.current_player]
        next_player = self.game.players[self.game.round.get_next_player()]
        known_cards = self.game.round.known_cards[0]
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

        extracted_state = {"obs": obs, "legal_actions": self._get_legal_actions()}

        if self.allow_raw_data:
            extracted_state["raw_obs"] = state
            extracted_state["raw_legal_actions"] = [a for a in state["legal_actions"]]

        if self.record_action:
            extracted_state["action_record"] = self.action_recorder

        return extracted_state

    def get_payoffs(self):
        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        if self.single_step:
            return utils.JOINED_ACTION_LIST[action_id]
        else:
            return utils.ACTION_LIST[action_id]

        # legal_ids = self._get_legal_actions()
        # if action_id in legal_ids:
        #     return utils.ACTION_LIST[action_id]
        # else:
        #     print("Tried non legal action", action_id, utils.ACTION_LIST[action_id], legal_ids, [utils.ACTION_LIST[a] for a in legal_ids])
        #     return utils.ACTION_LIST[np.random.choice(legal_ids)]

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        if self.game._single_step_actions:
            legal_ids = [utils.JOINED_ACTION_SPACE[action] for action in legal_actions]
        else:
            legal_ids = [utils.ACTION_SPACE[action] for action in legal_actions]

        return legal_ids

    def _load_model(self):
        """Load pretrained/rule model

        Returns:
            model (Model): A Model object
        """
        raise NotImplementedError

    def step(self, action, raw_action=False):
        if not raw_action:
            action = self._decode_action(action)
        if self.single_agent_mode:
            return self._single_agent_step(action)

        self.timestep += 1
        # Record the action for human interface
        if self.record_action:
            self.action_recorder.append([self.get_player_id(), action])
        next_state, player_id = self.game.step(action)

        return self._extract_state(next_state), player_id

    def run(self, is_training=False):
        """
        Run a complete game, either for evaluation or training RL agent.

        Args:
            is_training (boolean): True if for training purpose.

        Returns:
            (tuple) Tuple containing:

                (list): A list of trajectories generated from the environment.
                (list): A list payoffs. Each entry corresponds to one player.

        Note: The trajectories are 3-dimension list. The first dimension is for different players.
              The second dimension is for different transitions. The third dimension is for the contents of each transiton
        """
        if self.single_agent_mode:
            raise ValueError("Run in single agent not allowed.")

        trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.reset()

        # Loop to play the game
        trajectories[player_id].append(state)

        while not self.is_over():
            # Agent plays
            if is_training:
                action = self.agents[player_id].step(state)
            else:
                action, _ = self.agents[player_id].eval_step(state)

            # Environment steps
            next_state, next_player_id = self.step(
                action, self.agents[player_id].use_raw
            )
            # Save action
            trajectories[player_id].append(action)

            decoded_action = action
            if not self.agents[player_id].use_raw:
                decoded_action = self._decode_action(action)
            trajectories[player_id].append(
                self.reward_func(state, next_state, decoded_action)
            )

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.player_num):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        # Reorganize the trajectories
        trajectories = reorganize(trajectories, payoffs)

        return trajectories, payoffs


def reorganize(trajectories, payoffs):
    """Reorganize the trajectory to make it RL friendly

    Args:
        trajectory (list): A list of trajectories
        payoffs (list): A list of payoffs for the players. Each entry corresponds to one player

    Returns:
        (list): A new trajectories that can be fed into RL algorithms.

    """
    player_num = len(trajectories)
    new_trajectories = [[] for _ in range(player_num)]

    for player in range(player_num):
        for i in range(0, len(trajectories[player]) - 3, 3):
            transition = trajectories[player][i : i + 4].copy()
            if i == len(trajectories[player]) - 3:
                transition[2] = payoffs[player]
                transition.append(True)
            else:
                transition.append(False)

            new_trajectories[player].append(transition)
    return new_trajectories