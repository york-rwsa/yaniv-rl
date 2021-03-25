import numpy as np
from pettingzoo.utils import wrappers
from pettingzoo.classic.rlcard_envs.rlcard_base import RLCardBase
from rlcard.utils.utils import print_card
from yaniv_rl.envs import make
from gym import spaces


class YanivBase(RLCardBase):
    def __init__(self, name, num_players, obs_shape, config={}):
        self.name = name

        self.config = config
        self.env = make(name, self.config)

        if not hasattr(self, "agents"):
            self.agents = [f"player_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]

        dtype = self.env.reset()[0]["obs"].dtype
        if dtype == np.dtype(np.int64):
            self._dtype = np.dtype(np.int8)
        elif dtype == np.dtype(np.float64):
            self._dtype = np.dtype(np.float32)
        else:
            self._dtype = dtype

        self.observation_spaces = self._convert_to_dict(
            [
                spaces.Dict(
                    {
                        "observation": spaces.Box(
                            low=0.0, high=1.0, shape=obs_shape, dtype=self._dtype
                        ),
                        "action_mask": spaces.Box(
                            low=0,
                            high=1,
                            shape=(self.env.game.get_action_num(),),
                            dtype=np.int8,
                        ),
                    }
                )
                for _ in range(self.num_agents)
            ]
        )
        self.action_spaces = self._convert_to_dict(
            [
                spaces.Discrete(self.env.game.get_action_num())
                for _ in range(self.num_agents)
            ]
        )

    def seed(self, seed=None):
        self.env = make(self.name, config={**self.config, "seed": seed})


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(YanivBase):
    metadata = {"render.modes": ["human"], "name": "yaniv_v0"}

    def __init__(self, config={}):
        super().__init__("yaniv", 3, (266,), config)

    def observe(self, agent):
        obs = self.env.get_state(self._name_to_int(agent))
        observation = np.ravel(obs["obs"])[:266].astype(self._dtype)

        legal_moves = self.next_legal_moves
        action_mask = np.zeros(self.env.action_num, int)
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": observation, "action_mask": action_mask}

    def render(self, mode="human"):
        for player in self.possible_agents:
            state = self.env.game.round.players[self._name_to_int(player)].hand
            print("\n===== {}'s Hand =====".format(player))
            print_card([c.__str__()[::-1] for c in state])
        state = self.env.game.get_state(0)
        print("\n==== Top Discarded Card ====")
        print_card([c.__str__() for c in state["discard_pile"][-1]] if state else None)
        print("\n")