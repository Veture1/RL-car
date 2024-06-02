import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
import math
from collections import namedtuple
from utils import ReplayBuffer, plot_rewards, RunningMeanStd





class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def observe(self, state, action, next_state, reward):
        # RandomAgent does not learn from observations
        pass

    def select_action(self, state):
        return self.action_space.sample()

    def update(self):
        # RandomAgent does not update any parameters
        pass


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class DQNAgent:
    def __init__(self, policy_net, target_net, action_dim, gamma, epsilon_start, epsilon_end, epsilon_decay, buffer, batch_size, lr,device, optimizer, tau):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.steps_done = 0
        self.device = device
        self.tau = tau
        
        self.Q = policy_net
        self.target_Q = target_net
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.target_Q.eval()
        
        self.optimizer = optimizer(self.Q.parameters(), lr=lr, amsgrad=True)
        self.replay_buffer = buffer
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
    
    def observe(self, state, action, reward, next_state):
        self.replay_buffer.push(state, action, reward, next_state)
    
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.Q(state)
                action = q_values.argmax().item()
                # print(f"Policy net result: {q_values}, action: {action}")
        else:
            action = random.choice(range(self.action_dim))
        return action, eps_threshold
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.tensor(batch.reward, device=self.device)


        # get Q values from policy network
        q_values = self.Q(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        # get Q values from target network
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_Q(non_final_next_states).max(1)[0]
        expected_q_values = reward_batch + self.gamma * next_state_values
        # print("mean expected Q values: ", expected_q_values.mean())
        
        criterion = nn.SmoothL1Loss()
        # print(f"expected Q values: {expected_q_values}")
        loss = criterion(q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(self.Q.parameters(), 1000)
        self.optimizer.step()

        # update target network
        target_net_dict = self.target_Q.state_dict()
        policy_net_dict = self.Q.state_dict()
        for key in target_net_dict:
            target_net_dict[key] = self.tau * policy_net_dict[key] + (1 - self.tau) * target_net_dict[key]
        self.target_Q.load_state_dict(target_net_dict)


class RNDAgent:
    def __init__(self, gamma, policy_net, target_net, rnd_target_net, rnd_predictor_net, action_dim, epsilon_start, epsilon_end, epsilon_decay, buffer, batch_size, lr,device, optimizer, tau):
        self.device = device
        self.action_dim = action_dim

        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.steps_done = 0
        self.tau = tau    # update rate for target network
        
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.rnd_target_net = rnd_target_net
        self.rnd_predictor_net = rnd_predictor_net
        self.rnd_target_net.eval()
        
        self.optimizer = optimizer(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.rnd_optimizer = optimizer(self.rnd_predictor_net.parameters(), lr=lr, amsgrad=True)
        self.replay_buffer = buffer
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

        self.state_mean_std = RunningMeanStd(shape=(1,))
        self.rnd_mean_std = RunningMeanStd(shape=(1,))

    

    def observe(self, state, action, reward, next_state):
        self.replay_buffer.push(state, action, reward, next_state)
        self.state_mean_std.update(state.cpu().numpy())

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.argmax().item()
        else:
            action = random.choice(range(self.action_dim))
        return action, eps_threshold


    def compute_rnd_reward(self, state):
        state = (state - self.state_mean_std.mean) / self.state_mean_std.std
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            rnd_target = self.rnd_target_net(state)
        rnd_prediction = self.rnd_predictor_net(state)
        rnd_reward = (rnd_target - rnd_prediction).pow(2).mean()
        
        rnd_reward = (rnd_reward.cpu().detach().numpy() - self.rnd_mean_std.mean) / self.rnd_mean_std.std
        rnd_reward = np.clip(rnd_reward, -5, 5)[0]
        
        return rnd_reward


    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = self.transition(*zip(*transitions))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

        # get Q values from policy network
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_q_values = reward_batch + self.gamma * next_state_values

        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1000)
        self.optimizer.step()

        # update target network
        target_net_dict = self.target_net.state_dict()
        policy_net_dict = self.policy_net.state_dict()
        for key in target_net_dict:
            target_net_dict[key] = self.tau * policy_net_dict[key] + (1 - self.tau) * target_net_dict[key]
        self.target_net.load_state_dict(target_net_dict)


        # Update RND predictor network
        self.rnd_optimizer.zero_grad()
        # normalize the target values
        state_batch_norm = (state_batch.cpu().detach().numpy() - self.state_mean_std.mean) / self.state_mean_std.std
        rnd_target_values = self.rnd_target_net(torch.tensor(state_batch_norm, dtype=torch.float32, device=self.device))
        rnd_prediction_values = self.rnd_predictor_net(torch.tensor(state_batch_norm, dtype=torch.float32, device=self.device))
        rnd_loss = (rnd_target_values - rnd_prediction_values).pow(2).mean()
        self.rnd_mean_std.update([rnd_loss.cpu().detach().numpy()])
        rnd_loss.backward()
        self.rnd_optimizer.step()