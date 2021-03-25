from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from yaniv_rl.envs import zoo_yaniv
import supersuit as ss

env = zoo_yaniv.env()

model = PPO(
    CnnPolicy,
    env,
    verbose=3,
    gamma=0.99,
    n_steps=125,
    ent_coef=0.01,
    learning_rate=0.00025,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gae_lambda=0.95,
    n_epochs=4,
    clip_range=0.2,
    clip_range_vf=1,
)
model.learn(total_timesteps=2000000)
model.save("policy")

# Rendering

env = zoo_yaniv.env()
model = PPO.load("policy")

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs)[0] if not done else None
    env.step(act)
    env.render()