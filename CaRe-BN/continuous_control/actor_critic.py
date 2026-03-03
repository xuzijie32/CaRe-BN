import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import copy
import torch.nn as nn
import torch.nn.functional as F
import SAN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, policy):
        super(Critic, self).__init__()
        self.policy = policy

        if policy == "TD3":
            # Q1 architecture
            self.l1 = nn.Linear(state_dim + action_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, 1)

            # Q2 architecture

            self.l4 = nn.Linear(state_dim + action_dim, 256)
            self.l5 = nn.Linear(256, 256)
            self.l6 = nn.Linear(256, 1)
        
        if policy == "DDPG":
            self.l1 = nn.Linear(state_dim, 400)
            self.l2 = nn.Linear(400+action_dim, 300)
            self.l3 = nn.Linear(300, 1)
            
    def forward(self, state, action):
        if self.policy == "DDPG":
            q1 = F.relu(self.l1(state))
            sa = torch.cat([q1, action], 1)
            q1 = F.relu(self.l2(sa))
            q1 = self.l3(q1)

            return q1
        if self.policy == "TD3":
            sa = torch.cat([state, action], 1)
            q1 = F.relu(self.l1(sa))
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)

            q2 = F.relu(self.l4(sa))
            q2 = F.relu(self.l5(q2))
            q2 = self.l6(q2)
            return q1, q2

    def Q1(self, state, action):
        if self.policy == "DDPG":
            q1 = F.relu(self.l1(state))
            sa = torch.cat([q1, action], 1)
            q1 = F.relu(self.l2(sa))
            q1 = self.l3(q1)

            return q1
        if self.policy == "TD3":
            sa = torch.cat([state, action], 1)

            q1 = F.relu(self.l1(sa))
            q1 = F.relu(self.l2(q1))
            q1 = self.l3(q1)
            return q1


class DDPG(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            spiking_neurons,
            discount=0.99,
            tau=0.001,
            BN=True,
            recalibration_batchs=100,

    ):

        if spiking_neurons == 'ANN':
            self.actor = SAN.ANN_Actor(state_dim, action_dim, max_action).to(device)
        else:
            self.actor = SAN.SNN_Actor(state_dim, action_dim, max_action, spiking_neurons, BN=BN).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim, "DDPG").to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),weight_decay=0.01,lr=1e-3)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.re_calibration_batchs=recalibration_batchs
        self.BN = BN

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action = self.actor_target(next_state)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q = self.critic(state, action)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # Compute actor losse
        actor_loss = -self.critic.Q1(state, self.actor(state, update = True)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return float(critic_loss), float(-actor_loss)

    def re_calibration(self, replay_buffer, batch_size=256):
        if self.BN:
            for _ in range(self.re_calibration_batchs):
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
                self.actor(state, re_calibration=[_, self.re_calibration_batchs])

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            spiking_neurons,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            BN=True,
            recalibration_batchs=100,
    ):

        if spiking_neurons == 'ANN':
            self.actor = SAN.ANN_Actor(state_dim, action_dim, max_action, BN=BN).to(device)
        else:
            self.actor = SAN.SNN_Actor(state_dim, action_dim, max_action, spiking_neurons, BN=BN).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, "TD3").to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.BN = BN
        self.re_calibration_batchs = recalibration_batchs

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state, update=True)).mean()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def re_calibration(self,replay_buffer, batch_size=256):
        if self.BN:
            for _ in range(self.re_calibration_batchs):
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
                self.actor(state,re_calibration=[_,self.re_calibration_batchs])
    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

