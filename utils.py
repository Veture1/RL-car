import matplotlib.pyplot as plt
import matplotlib.animation
import torch

import time
import IPython
import numpy as np

from collections import deque, namedtuple
import random


def plot_rewards(episode_rewards, episode_steps, realtime=False):
    fig = plt.figure(figsize=(12, 8))
    rewards = torch.tensor(episode_rewards, dtype=torch.float)
    steps = torch.tensor(episode_steps, dtype=torch.int)
    
    if realtime:
        plt.clf()
        plt.suptitle('Training...')
    else:
        plt.suptitle('Training Result')

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_ylabel('Reward')
    ax1.grid(linestyle='--')
    ax1.tick_params(axis='x', length=0)
    ax1.plot(rewards.numpy(), label='Episode Reward')

    if len(rewards) >= 100:
        bin_size = 100
    else:
        bin_size = len(rewards)
    means = rewards.unfold(0, bin_size, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(bin_size - 1) + means[0], means))
    ax1.plot(means.numpy(), label='Average Reward (100 episodes)')
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(linestyle='--')
    ax2.plot(steps.numpy(), label='Episode Steps')
    ax2.legend()

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.10)
    
    plt.pause(0.001)
    if realtime :
        IPython.display.display(plt.gcf())
        IPython.display.clear_output(wait=True)
    else :
        IPython.display.display(plt.gcf())


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.transtion = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(self.transtion(*args))
    
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, x):
        x = np.array(x, dtype='float32')
        if x.ndim == 0:
            x = x.reshape(1)
        elif x.ndim == 1 and x.shape[0] == 1:
            x = x.reshape(-1)

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)
