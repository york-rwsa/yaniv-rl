import json
from yaniv_rl import utils
import yaml

import numpy as np
from yaniv_rl.envs.rllib_multiagent_yaniv import YanivEnv
from yaniv_rl.models.yaniv_rule_models import YanivNoviceRuleAgent

from copy import deepcopy
from ray.tune.logger import pretty_print


class YanivTournament:
    def __init__(self, env_config, trainers=[]):
        self.env_config = env_config
        self.trainers = trainers

        self.rule_agent = YanivNoviceRuleAgent(
            single_step=env_config.get("single_step", True)
        )

        self.env = YanivEnv(env_config)

        self.players = []
        for i in range(self.env.num_players):
            if i < len(self.trainers):
                self.players.append(self.trainers[i])
            else:
                self.players.append(self.rule_agent)

    def run_episode(self):
        obs = self.env.reset()
        done = {"__all__": False}

        steps = 0
        while not done["__all__"]:
            player = self.players[self.env.current_player]
            player_id = self.env.current_player_string

            if player in self.trainers:
                action = player.compute_action(obs[player_id], policy_id="policy_1")

                if self.env.game.round.discarding:
                    dec_action = self.env._decode_action(action)
                    if dec_action != utils.YANIV_ACTION:
                        self.player_stats[player_id]["discard_freqs"][
                            str(int(len(dec_action) / 2)) 
                        ] += 1
                else:
                    pickup_action = self.env._decode_action(action)
                    self.player_stats[player_id]["pickup_freqs"][pickup_action] += 1
                    
                obs, reward, done, info = self.env.step({player_id: action})
            else:
                state = self.env.game.get_state(self.env.current_player)
                extracted_state = {}
                extracted_state["raw_obs"] = state
                extracted_state["raw_legal_actions"] = [
                    a for a in state["legal_actions"]
                ]
                action = self.rule_agent.step(extracted_state)

                if self.env.game.round.discarding:
                    if action != utils.YANIV_ACTION:
                        self.player_stats[player_id]["discard_freqs"][
                            str(int(len(action) / 2)) 
                        ] += 1
                else:
                    self.player_stats[player_id]["pickup_freqs"][action] += 1
                    

                obs, reward, done, info = self.env.step(
                    {player_id: action}, raw_action=True
                )

            steps += 1

        self.game_stats["avg_roundlen"] += steps

        winner = self.env.game.round.winner
        if winner == -1:
            self.game_stats["avg_draws"] += 1
        else:
            winner_id = self.env._get_player_string(winner)
            self.player_stats[winner_id]["avg_wins"] += 1
            self.player_stats[winner_id]["winning_hands"].append(
                utils.get_hand_score(self.env.game.players[winner].hand)
            )

        assaf = self.env.game.round.assaf
        if assaf is not None:
            self.player_stats[self.env._get_player_string(assaf)]["avg_assafs"] += 1

        s = self.env.game.round.scores
        if s is not None:
            for i in range(self.env.num_players):
                if s[i] > 0:
                    self.player_stats[self.env._get_player_string(i)]["scores"].append(
                        s[i]
                    )

        self.games_played += 1

    def run(self, eval_num):
        self.reset_stats()

        for _ in range(eval_num):
            self.run_episode()

        return self.get_average_stats()

    def reset_stats(self):
        self.games_played = 0

        self.game_stats = {
            "avg_roundlen": 0,
            "avg_draws": 0,
        }

        self.player_stats = {
            player_id: {
                "avg_wins": 0,
                "avg_assafs": 0,
                "scores": [],
                "winning_hands": [],
                "discard_freqs": {
                    "1": 0,
                    "2": 0,
                    "3": 0,
                    "4": 0,
                    "5": 0,
                },
                "pickup_freqs": {
                    a: 0 for a in utils.pickup_actions
                }
            }
            for player_id in self.env._get_players()
        }

    def get_average_stats(self):
        stats = {
            "game": deepcopy(self.game_stats),
            "player": deepcopy(self.player_stats),
        }

        for key in stats["game"].keys():
            if key.startswith("avg"):
                stats["game"][key] /= self.games_played

        for player_stats in stats["player"].values():
            for key in player_stats:
                if key.startswith("avg"):
                    player_stats[key] /= self.games_played

            player_stats["avg_losing_score"] = (
                np.mean(player_stats["scores"])
                if len(player_stats["scores"]) > 0
                else 0
            )
            player_stats.pop("scores")

            player_stats["avg_winning_hand"] = (
                np.mean(player_stats["winning_hands"])
                if len(player_stats["winning_hands"]) > 0
                else 0
            )
            player_stats.pop("winning_hands")

        return stats

    def print_stats(self):
        avg_stats = self.get_average_stats()
        cleaned = json.dumps(avg_stats)

        print(yaml.safe_dump(json.loads(cleaned), default_flow_style=False))
