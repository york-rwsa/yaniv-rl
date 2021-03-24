import torch

import matplotlib.pyplot as plt
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils.utils import set_global_seed
from rlcard.utils import Logger
from yaniv_rl.utils import tournament
from rlcard.agents import NFSPAgentPytorch as NFSPAgent
yaniv_rl.models.yaniv_rule_models import YanivNoviceRuleAgent

plt.style.use("ggplot")

config = {
    "end_after_n_deck_replacements": 0,
    "end_after_n_steps": 500,
    "early_end_reward": 0,
    "use_scaled_negative_reward": False,
    "seed": 0,
}
memory_init_size = 1000

env = rlcard.make("yaniv", config=config)

modelfiles = [
    "yaniv_nfsp/20210320_vs_random/model/model.pth",
    "yaniv_nfsp/20210320_1950/model/model_1.pth",
]

agents = []
for m in modelfiles:    
    agent = NFSPAgent(
        scope="nfsp1",
        action_num=env.action_num,
        state_shape=env.state_shape,
        hidden_layers_sizes=[512, 1024, 2048, 1024, 512],
        anticipatory_param=0.5,
        batch_size=256,
        train_every=0,
        rl_learning_rate=0.00005,
        sl_learning_rate=0.00001,
        min_buffer_size_to_learn=memory_init_size,
        q_replay_memory_size=int(1e5),
        q_replay_memory_init_size=memory_init_size,
        q_train_every=0,
        q_batch_size=256,
        q_mlp_layers=[512, 1024, 2048, 1024, 512],
        device=torch.device("cuda"),
    )

    statedict = torch.load(m)
    agent.load(statedict)
    agents.append(agent)

env.set_agents(agents)

payoffs, wins, draws, roundlen = tournament(env, 1000)

print("Timestep: {}, avg roundlen: {}".format(env.timestep, roundlen))
for i in range(env.player_num):
    print(
        "Agent {}:\nWins: {}, Draws: {}, Payoff: {}".format(
            i, wins[i], draws, payoffs[i]
        )
    )
