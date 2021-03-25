import yaniv_rl.envs.gym_yaniv
import regym

from regym.environments import generate_task, EnvType
from regym.rl_algorithms import build_Random_Agent, build_PPO_Agent
import numpy as np
import random
import gym_kuhn_poker

hyperparams = {
    "actor_arch": 'None',
    "adam_eps": 1.0e-05,
    "critic_arch": 'None',
    "discount": 0.99,
    "entropy_weight": 0.01,
    "gae_tau": 0.95,
    "gradient_clip": 5,
    "horizon": 128,
    "learning_rate": 0.0003,
    "mini_batch_size": 16,
    "optimization_epochs": 10,
    "phi_arch": "MLP",
    "ppo_ratio_clip": 0.2,
    "use_cuda": False,
    "use_gae": 'true',
}

def main():
    task = generate_task("Yaniv-v0", EnvType.MULTIAGENT_SEQUENTIAL_ACTION)
    # random_r1 = build_Random_Agent(task, {}, agent_name="random")

    ppo = build_PPO_Agent(task, hyperparams, "ppo")

    traj = task.run_episode([ppo, ppo], training=True,)
    print(traj)


if __name__ == "__main__":
    main()