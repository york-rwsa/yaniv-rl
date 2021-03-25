# adapted from https://github.com/ChenglongChen/pytorch-DRL/blob/master/A2C.py

import os
from rlcard.utils.utils import remove_illegal
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop
from torch.autograd import Variable

from collections import namedtuple
import random
import numpy as np

import wandb


class A2CAgentPytorch(object):
    """
    A unified agent interface:
    - interact: interact with the environment to collect experience
        - _take_one_step: take one step
        - _take_n_steps: take n steps
        - _discount_reward: discount roll out rewards
    - train: train on a sample batch
        - _soft_update_target: soft update the target network
    - exploration_action: choose an action based on state with random noise
                            added for exploration in training
    - action: choose an action based on state for execution
    - value: evaluate value for a state-action pair
    - evaluation: evaluation a learned agent
    """

    def __init__(
        self,
        state_shape,
        action_dim,
        memory_capacity=10000,
        max_steps=None,
        reward_gamma=0.95,
        reward_scale=1.0,
        done_penalty=None,
        actor_hidden_size=128,
        critic_hidden_size=128,
        actor_output_act=nn.functional.log_softmax,
        critic_loss="mse",
        actor_lr=0.001,
        critic_lr=0.001,
        optimizer_type="adam",
        entropy_reg=0.01,
        max_grad_norm=0.5,
        batch_size=100,
        epsilon_start=0.9,
        epsilon_end=0.01,
        epsilon_decay=0.99,
        use_cuda=True,
        train_every_n_episodes=1,
    ):
        self.use_raw = False

        self.state_dim = np.prod(state_shape)
        self.action_dim = action_dim
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.memory = ReplayMemory(memory_capacity)
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.train_every_n_episodes = train_every_n_episodes
        self.target_tau = 0.01

        # params for epsilon greedy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.use_cuda = use_cuda and th.cuda.is_available()

        self.actor = ActorNetwork(
            self.state_dim,
            self.actor_hidden_size,
            self.action_dim,
            self.actor_output_act,
        )
        self.critic = CriticNetwork(
            self.state_dim, self.action_dim, self.critic_hidden_size, 1
        )
        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)
        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()

    # def resetMemory(self):
    #     self.states = []
    #     self.actions = []
    #     self.rewards = []

    # def feed(self, ts):
    #     (state, action, reward, next_state, done) = tuple(ts)
    #     obs = np.ravel(state['obs'])
    #     self.memory._push_one(obs, action, reward)
    #     self.n_steps += 1

    #     if done:
    #         self.n_episodes += 1

    #     if self.n_steps % self.train_every == 0:
    #         self.train()

    def feed_game(self, trajectories):
        t = list(zip(*trajectories))

        states = [np.ravel(s["obs"]) for s in t[0]]
        actions = t[1]
        rewards = t[2]

        self.n_episodes += 1

        rewards = self._discount_reward(rewards, 0)
        self.n_steps += len(trajectories)

        self.memory.push(states, actions, rewards)

        if self.n_episodes % self.train_every_n_episodes == 0:
            self.train()

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    # soft update the actor target network or critic target network
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_((1.0 - self.target_tau) * t.data + self.target_tau * s.data)

    # train on a roll out batch
    def train(self):
        batch = self.memory.sample(self.batch_size)
        one_hot_actions = index_to_one_hot(batch.actions, self.action_dim)

        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        actions_var = to_tensor_var(one_hot_actions, self.use_cuda).view(
            -1, self.action_dim
        )
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)

        # update actor network
        self.actor_optimizer.zero_grad()
        action_log_probs = self.actor(states_var)
        entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
        action_log_probs = th.sum(action_log_probs * actions_var, 1)
        values = self.critic(states_var, actions_var)
        advantages = rewards_var - values.detach()
        pg_loss = -th.mean(action_log_probs * advantages)
        actor_loss = pg_loss - entropy_loss * self.entropy_reg
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        self.critic_optimizer.zero_grad()
        target_values = rewards_var
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # print("actor loss {}, critic loss {}".format(actor_loss, critic_loss))
        wandb.log(
            {
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "total_loss": actor_loss + critic_loss,
            }
        )

    # predict softmax action based on state
    def _softmax_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        softmax_action_var = th.exp(self.actor(state_var))
        if self.use_cuda:
            softmax_action = softmax_action_var.data.cpu().numpy()[0]
        else:
            softmax_action = softmax_action_var.data.numpy()[0]
        return softmax_action

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state, legal_actions):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1.0 * self.n_steps * self.epsilon_decay
        )
        if np.random.rand() < epsilon:
            action = np.random.choice(legal_actions)
        else:
            softmax_action = self._softmax_action(state)
            action = self._pick_action(softmax_action, legal_actions)

        return action

    # choose an action based on state for execution
    def action(self, state, legal_actions):
        softmax_action = self._softmax_action(state)
        return self._pick_action(softmax_action, legal_actions)

    def _pick_action(self, softmax_action, legal_actions):
        # ie if all probs are 0
        legal_probs = remove_illegal(softmax_action, legal_actions)
        if not np.any(legal_probs):
            action = np.random.choice(legal_actions)
        else:
            action = np.argmax(legal_probs)

        return action

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action], self.use_cuda)
        value_var = self.critic(state_var, action_var)
        if self.use_cuda:
            value = value_var.data.cpu().numpy()[0]
        else:
            value = value_var.data.numpy()[0]
        return value

    def step(self, state):
        obs = np.ravel(state["obs"])
        return self.exploration_action(obs, state["legal_actions"])

    def eval_step(self, state):
        obs = np.ravel(state["obs"])
        softmax_action = remove_illegal(
            self._softmax_action(obs), state["legal_actions"]
        )
        action = np.argmax(softmax_action)
        return action, softmax_action

    def save(self, dir):
        th.save(self.actor.state_dict(), os.path.join(dir, "actor.pth"))
        th.save(self.critic.state_dict(), os.path.join(dir, "critic.pth"))

    def load(self, dir):
        self.actor.load_state_dict(th.load(os.path.join(dir, "actor.pth")))
        self.critic.load_state_dict(th.load(os.path.join(dir, "critic.pth")))

        self.actor.eval()
        self.critic.eval()


class ActorNetwork(nn.Module):
    """
    A network for actor
    """

    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc21 = nn.Linear(hidden_size*2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # activation function for the output
        self.output_act = output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        # out = nn.functional.relu(self.fc21(out))
        out = self.output_act(self.fc3(out))
        return out


class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def __call__(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        out = th.cat([out, action], 1)
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def index_to_one_hot(index, dim):
    if isinstance(index, np.int) or isinstance(index, np.int64):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.0
    else:
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.0
    return one_hot


def to_tensor_var(x, use_cuda=True, dtype="float"):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(LongTensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))


def entropy(p):
    return -th.sum(p * th.log(p), 1)


Experience = namedtuple(
    "Experience", ("states", "actions", "rewards", "next_states", "dones")
)


class ReplayMemory(object):
    """
    Replay memory buffer
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, next_states=None, dones=None):
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s, a, r, n_s, d in zip(
                    states, actions, rewards, next_states, dones
                ):
                    self._push_one(s, a, r, n_s, d)
            else:
                for s, a, r in zip(states, actions, rewards):
                    self._push_one(s, a, r)
        else:
            self._push_one(states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)
