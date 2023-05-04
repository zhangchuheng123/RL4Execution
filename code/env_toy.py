"""
A toy trade execution environment and possible DRL solutions.
"""


import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import warnings
warnings.simplefilter("error")

from pathos.multiprocessing import ProcessingPool as Pool
from collections import deque
from tqdm import trange
import pandas as pd
import numpy as np
import itertools
import pdb
import os


EPS = 1e-5


class DefaultConfig(object):

    random_sigma_func = staticmethod(lambda size: np.random.uniform(1, 2, size=size))
    random_alpha_func = staticmethod(lambda size: np.random.normal(0, 0.5, size=size))

    result_path = 'results/exp43_new'
    batch_size = 128
    num_iterations = 1000000
    total_data = 10000
    state_length = 30
    future_length = 30
    total_length = state_length + future_length
    actor_lr = 1e-4
    critic_lr = 1e-4
    encoder_lr = 1e-4
    noise = 1.0
    entropy_coeff = 1.0
    gamma = np.exp(np.log(0.5) / 30)
    device = 'cuda'

    action_emb_dim = 2
    state_emb_dim = 2

    bottleneck_mode = 'shared'
    pretrain = False
    label_std_mapping = True
    label_noise = 5.0


class EncoderMLPNetwork(nn.Module):
    def __init__(self, dim_input, dim_output, hidden=128):
        super(EncoderMLPNetwork, self).__init__()

        self.fc1 = nn.Linear(dim_input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, dim_output)

    def forward(self, inputs):

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ActorEmbMLPNetwork(nn.Module):
    def __init__(self, dim_input, dim_output, hidden=128):
        super(ActorEmbMLPNetwork, self).__init__()

        self.fc1 = nn.Linear(dim_input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, dim_output)
        self.output_activation = nn.Softmax(dim=1)

    def forward(self, embedings, noise=None, require_logits=False):
        x = F.relu(self.fc1(embedings))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        if noise is not None:
            logits = logits \
                + torch.normal(0, noise, size=logits.shape, device=logits.device)
        probs = self.output_activation(logits)
        if require_logits:
            return probs, logits
        else:
            return probs


class CriticEmbMLPNetwork(nn.Module):
    def __init__(self, dim_input1, dim_input2, dim_output=1, hidden=128):
        super(CriticEmbMLPNetwork, self).__init__()

        self.fc1 = nn.Linear(dim_input1 + dim_input2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, dim_output)

    def forward(self, emb_states, emb_actions):
        x = torch.cat((emb_states, emb_actions), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorMLPNetwork(nn.Module):
    def __init__(self, dim_input, dim_output, hidden=128):
        super(ActorMLPNetwork, self).__init__()
        self.fc1 = nn.Linear(dim_input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, dim_output)
        self.output_activation = nn.Softmax(dim=1)
        
    def forward(self, inputs, noise=None, require_logits=False):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        if noise is None:
            logits = self.fc3(x)
            probs = self.output_activation(logits)
        else:
            x = self.fc3(x)
            logits = x + torch.normal(0, noise, size=x.shape, device=x.device)
            probs = self.output_activation(logits)

        if require_logits:
            return probs, logits
        else:
            return probs


class CriticMLPNetwork(nn.Module):
    def __init__(self, dim_input, dim_output, hidden=128):
        super(CriticMLPNetwork, self).__init__()
        self.fc1 = nn.Linear(dim_input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, dim_output)
        
    def forward(self, states, actions):
        x = torch.cat((states, actions), 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DecoderMLPNetwork(nn.Module):
    def __init__(self, dim_state, dim_emb=2, hidden=128):
        super(DecoderMLPNetwork, self).__init__()
        self.fc1 = nn.Linear(dim_emb, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, dim_state)

    def forward(self, embedings):
        x = F.relu(self.fc1(embedings))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent(object):
    def __init__(self, config):
        self.config = config
        self.dim_state = self.config.state_length
        self.dim_action = self.config.future_length
        self._set_seed()

    def _set_seed(self, seed=None):
        if seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder='little')
        else:
            seed = seed + 1234
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _to_tensor(self, tensor, dtype=torch.float):
        return torch.tensor(tensor, dtype=dtype, device=self.config.device)

    def generate_data(self, size=None):
        if size is None:
            size = self.config.batch_size

        sigmas = self.config.random_sigma_func((size, 1))
        alphas = self.config.random_alpha_func((size, 1))
        data = (np.random.standard_normal(size=(size, self.config.total_length)) * sigmas + alphas)
        return data, sigmas, alphas

    def get_reward(self, future_states, actions):
        future_prices = torch.cumsum(future_states, dim=1)
        gammas = torch.pow(self.config.gamma, 
            torch.arange(self.dim_action, device=self.config.device)).view(1, -1)
        baseline_action = torch.ones(actions.shape, device=self.config.device) / self.dim_action
        reward = future_prices * (actions - baseline_action) * gammas
        return reward.sum(1, keepdims=True)

    def evaluate(self, actor, train=False):
        if train:
            inds = np.random.choice(self.config.total_data, self.config.batch_size)
            data = self.data[inds]
        else:
            data, _, _ = self.generate_data()
            data = self._to_tensor(data)

        states = data[:, :self.config.state_length]
        future_states = data[:, -self.config.future_length:]

        actions = actor(states)
        rewards = self.get_reward(future_states, actions)

        return dict(reward_mean=float(rewards.mean()), reward_std=float(rewards.std()))

    def evaluate_sample(self, actor, filename):
        sigmas = np.array([[1.5], [1.5], [1.5], [1.5], [1.5]])
        alphas = np.array([[1.0], [0.5], [0.0], [-0.5], [-1.0]])
        data = (np.random.standard_normal(size=(5, self.config.total_length)) * sigmas + alphas)
        prices = np.cumsum(data, axis=1)

        states = data[:, :self.config.state_length]
        future_states = data[:, -self.config.future_length:]
        states = self._to_tensor(states)
        future_states = self._to_tensor(future_states)
        actions = actor(states)

        rewards = self.get_reward(future_states, actions)
        actions = actions.detach().cpu().numpy()

        rate_max = np.max(actions) + 0.01
        fontsize = 15

        plt.figure(figsize=(25, 5))

        ax1 = plt.subplot(151)
        ax2 = ax1.twinx()
        ax1.plot(np.arange(self.config.total_length), prices[0, :], 'C0', lw=3)
        ax2.plot(np.arange(self.config.future_length) + self.config.state_length, actions[0, :], 'C1', lw=3)
        ax1.set_title('Sharply Rising', fontsize=fontsize)
        ax1.set_ylabel('Price', fontsize=fontsize)
        ax1.set_xlabel('Time', fontsize=fontsize)
        ax2.set_ylim([0, np.max(actions[0, :]) + 0.01])
        ax1.yaxis.set_major_locator(ticker.NullLocator())
        # ax2.yaxis.set_major_locator(ticker.NullLocator())

        ax1 = plt.subplot(152)
        ax2 = ax1.twinx()
        ax1.plot(np.arange(self.config.total_length), prices[1, :], 'C0', lw=3)
        ax2.plot(np.arange(self.config.future_length) + self.config.state_length, actions[1, :], 'C1', lw=3)
        ax1.set_title('Slowly Rising', fontsize=fontsize)
        ax1.set_xlabel('Time', fontsize=fontsize)
        ax2.set_ylim([0, np.max(actions[1, :]) + 0.01])
        ax1.yaxis.set_major_locator(ticker.NullLocator())
        # ax2.yaxis.set_major_locator(ticker.NullLocator())

        ax1 = plt.subplot(153)
        ax2 = ax1.twinx()
        ax1.plot(np.arange(self.config.total_length), prices[2, :], 'C0', lw=3)
        ax2.plot(np.arange(self.config.future_length) + self.config.state_length, actions[2, :], 'C1', lw=3)
        ax1.set_title('Fluctuating', fontsize=fontsize)
        ax1.set_xlabel('Time', fontsize=fontsize)
        ax2.set_ylim([0, np.max(actions[2, :]) + 0.01])
        ax1.yaxis.set_major_locator(ticker.NullLocator())
        # ax2.yaxis.set_major_locator(ticker.NullLocator())

        ax1 = plt.subplot(154)
        ax2 = ax1.twinx()
        ax1.plot(np.arange(self.config.total_length), prices[3, :], 'C0', lw=3)
        ax2.plot(np.arange(self.config.future_length) + self.config.state_length, actions[3, :], 'C1', lw=3)
        ax1.set_title('Slowly Decreasing', fontsize=fontsize)
        ax1.set_xlabel('Time', fontsize=fontsize)
        ax2.set_ylim([0, np.max(actions[3, :]) + 0.01])
        ax1.yaxis.set_major_locator(ticker.NullLocator())
        # ax2.yaxis.set_major_locator(ticker.NullLocator())

        ax1 = plt.subplot(155)
        ax2 = ax1.twinx()
        ax1.plot(np.arange(self.config.total_length), prices[4, :], 'C0', lw=3)
        ax2.plot(np.arange(self.config.future_length) + self.config.state_length, actions[4, :], 'C1', lw=3)
        ax1.set_title('Sharply Decreasing', fontsize=fontsize)
        ax1.set_xlabel('Time', fontsize=fontsize)
        ax2.set_ylim([0, np.max(actions[4, :]) + 0.01])
        ax1.yaxis.set_major_locator(ticker.NullLocator())
        ax2.set_ylabel('Trading Rate', fontsize=fontsize)

        plt.savefig(filename, bbox_inches='tight')
        plt.close('all')


    def evaluate_sample_sigma_alpha(self, actor, filename):
        sigmas, alphas = np.meshgrid([0.5, 1.0, 1.5, 2.0, 3.0], [-1.0, -0.5, 0.0, 0.5, 1.0])
        sigmas = sigmas.reshape((-1, 1))
        alphas = alphas.reshape((-1, 1))
        data = (np.random.standard_normal(size=(25, self.config.total_length)) * sigmas + alphas)
        prices = np.cumsum(data, axis=1)

        states = data[:, :self.config.state_length]
        future_states = data[:, -self.config.future_length:]
        states = self._to_tensor(states)
        future_states = self._to_tensor(future_states)
        actions = actor(states)

        rewards = self.get_reward(future_states, actions)
        actions = actions.detach().cpu().numpy()

        plt.figure(figsize=(25, 25))
        for i in range(5):
            for j in range(5):
                ind = i * 5 + j
                plt.subplot(5, 5, ind + 1)
                plt.plot(actions[ind, :])
                plt.title('alpha={:.1f} sigma={:.1f}'.format(alphas[ind, 0], sigmas[ind, 0]))

        plt.savefig(filename, bbox_inches='tight')
        plt.close('all')


    def evaluate_sample_IS(self, actor, filename, csv_filename=None):
        inds = np.random.choice(self.config.total_data, 5)
        data = self.data[inds]
        sigmas = self.sigmas[inds]
        alphas = self.alphas[inds]
        prices = np.cumsum(data.cpu().numpy(), axis=1)

        states = data[:, :self.config.state_length]
        future_states = data[:, -self.config.future_length:]
        actions = actor(states)

        rewards = self.get_reward(future_states, actions)
        actions = actions.detach().cpu().numpy()

        plt.figure(figsize=(25, 5))

        ax1 = plt.subplot(151)
        ax2 = ax1.twinx()
        ax1.plot(np.arange(self.config.total_length), prices[0, :], 'C0')
        ax2.plot(np.arange(self.config.future_length) + self.config.state_length, actions[0, :], 'C1')
        ax2.set_ylim([0, np.max(actions[0, :]) + 0.01])
        ax1.set_title('reward={:.4f} max={}'.format(float(rewards[0]), np.argmax(actions[0, :])))

        ax1 = plt.subplot(152)
        ax2 = ax1.twinx()
        ax1.plot(np.arange(self.config.total_length), prices[1, :], 'C0')
        ax2.plot(np.arange(self.config.future_length) + self.config.state_length, actions[1, :], 'C1')
        ax2.set_ylim([0, np.max(actions[1, :]) + 0.01])
        ax1.set_title('reward={:.4f} max={}'.format(float(rewards[1]), np.argmax(actions[1, :])))

        ax1 = plt.subplot(153)
        ax2 = ax1.twinx()
        ax1.plot(np.arange(self.config.total_length), prices[2, :], 'C0')
        ax2.plot(np.arange(self.config.future_length) + self.config.state_length, actions[2, :], 'C1')
        ax2.set_ylim([0, np.max(actions[2, :]) + 0.01])
        ax1.set_title('reward={:.4f} max={}'.format(float(rewards[2]), np.argmax(actions[2, :])))

        ax1 = plt.subplot(154)
        ax2 = ax1.twinx()
        ax1.plot(np.arange(self.config.total_length), prices[3, :], 'C0')
        ax2.plot(np.arange(self.config.future_length) + self.config.state_length, actions[3, :], 'C1')
        ax2.set_ylim([0, np.max(actions[3, :]) + 0.01])
        ax1.set_title('reward={:.4f} max={}'.format(float(rewards[3]), np.argmax(actions[3, :])))

        ax1 = plt.subplot(155)
        ax2 = ax1.twinx()
        ax1.plot(np.arange(self.config.total_length), prices[4, :], 'C0')
        ax2.plot(np.arange(self.config.future_length) + self.config.state_length, actions[4, :], 'C1')
        ax2.set_ylim([0, np.max(actions[4, :]) + 0.01])
        ax1.set_title('reward={:.4f} max={}'.format(float(rewards[4]), np.argmax(actions[4, :])))

        plt.savefig(filename, bbox_inches='tight')
        plt.close('all')

        if csv_filename is not None:
            pd.DataFrame(np.concatenate((prices, actions), axis=1)).to_csv(csv_filename)


class AgentPredictFull(Agent):
    def __init__(self, config):
        super(AgentPredictFull, self).__init__(config)

        self.action_encoder = EncoderMLPNetwork(dim_input=self.dim_action, dim_output=self.config.action_emb_dim)\
            .to(device=self.config.device)
        self.state_encoder = EncoderMLPNetwork(dim_input=self.dim_state, dim_output=self.config.state_emb_dim)\
            .to(device=self.config.device)
        self.actor = ActorEmbMLPNetwork(dim_input=self.config.state_emb_dim, dim_output=self.dim_action)\
            .to(device=self.config.device)
        self.critic = CriticEmbMLPNetwork(dim_input1=self.config.state_emb_dim, dim_input2=self.config.action_emb_dim)\
            .to(device=self.config.device)
        self.decoder = DecoderMLPNetwork(dim_state=self.dim_state)\
            .to(device=self.config.device)

        critic_params = list(self.critic.parameters()) + list(self.action_encoder.parameters())
        encoder_params = list(self.state_encoder.parameters()) + list(self.decoder.parameters())
        self.actor_optim = opt.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optim = opt.Adam(critic_params, lr=self.config.critic_lr)
        self.encoder_optim = opt.Adam(encoder_params, lr=self.config.encoder_lr)

        data, sigmas, alphas = self.generate_data(self.config.total_data)
        self.data = self._to_tensor(data)
        self.sigmas = self._to_tensor(sigmas)
        self.alphas = self._to_tensor(alphas)

    def learn(self):

        info = []
        actor_func = lambda states: self.actor(self.state_encoder(states))

        for i in trange(self.config.num_iterations):

            # Step 1: sample a batch
            inds = np.random.choice(self.config.total_data, self.config.batch_size)
            data = self.data[inds]
            sigmas = self.sigmas[inds]
            alphas = self.alphas[inds]
            states = data[:, :self.config.state_length]
            future_states = data[:, -self.config.future_length:]

            # Step 2: learn encoder
            loss_encoder = nn.MSELoss()(states, self.decoder(self.state_encoder(states)))
            self.encoder_optim.zero_grad()
            loss_encoder.backward()
            self.encoder_optim.step()

            # Step 3: learn critic
            actions = self.actor(self.state_encoder(states), noise=self.config.noise).detach()
            rewards = self.get_reward(future_states, actions).detach()
            values = self.critic(self.state_encoder(states), self.action_encoder(actions))
            loss_critic = nn.MSELoss()(values, rewards)
            self.critic_optim.zero_grad()
            loss_critic.backward()
            self.critic_optim.step()

            # Step 4: learn actor
            actions, logits = self.actor(self.state_encoder(states), require_logits=True)
            loss_actor = - torch.mean(self.critic(self.state_encoder(states), self.action_encoder(actions))) \
                - self.config.entropy_coeff * Categorical(logits=logits).entropy().mean()
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            if i % 1000 == 0:
                e_eval = self.evaluate(actor_func)
                e_train = self.evaluate(actor_func, train=True)
                info.append(dict(
                    iterations=i,
                    loss_encoder=float(loss_encoder),
                    loss_actor=float(loss_actor),
                    loss_critic=float(loss_critic),
                    loss_total=float(loss_actor) + float(loss_critic) + float(loss_encoder),
                    reward_mean_train=e_train['reward_mean'],
                    reward_std_train=e_train['reward_std'],
                    reward_mean_eval=e_eval['reward_mean'],
                    reward_std_eval=e_eval['reward_std'],
                ))

            if i % 10000 == 0:
                pd.DataFrame(info).to_csv(os.path.join(self.config.result_path, 'result_tmp.csv'))
                filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}.png'.format(i))
                self.evaluate_sample(actor_func, filename)

        return info


class AgentPredictParam(Agent):
    def __init__(self, config):
        super(AgentPredictParam, self).__init__(config)

        self.action_encoder = EncoderMLPNetwork(dim_input=self.dim_action, dim_output=self.config.action_emb_dim)\
            .to(device=self.config.device)
        self.state_encoder = EncoderMLPNetwork(dim_input=self.dim_state, dim_output=self.config.state_emb_dim)\
            .to(device=self.config.device)
        self.actor = ActorEmbMLPNetwork(dim_input=self.config.state_emb_dim, dim_output=self.dim_action)\
            .to(device=self.config.device)
        self.critic = CriticEmbMLPNetwork(dim_input1=self.config.state_emb_dim, dim_input2=self.config.action_emb_dim)\
            .to(device=self.config.device)

        critic_params = list(self.critic.parameters()) + list(self.action_encoder.parameters())
        self.actor_optim = opt.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optim = opt.Adam(critic_params, lr=self.config.critic_lr)
        self.encoder_optim = opt.Adam(self.state_encoder.parameters(), lr=self.config.encoder_lr)

        data, sigmas, alphas = self.generate_data(self.config.total_data)
        self.data = self._to_tensor(data)
        self.sigmas = self._to_tensor(sigmas)
        self.alphas = self._to_tensor(alphas)

    def learn(self):

        info = []
        actor_func = lambda states: self.actor(self.state_encoder(states))

        for i in trange(self.config.num_iterations):

            # Step 1: sample a batch
            inds = np.random.choice(self.config.total_data, self.config.batch_size)
            data = self.data[inds]
            sigmas = self.sigmas[inds]
            alphas = self.alphas[inds]

            states = data[:, :self.config.state_length]
            future_states = data[:, -self.config.future_length:]

            # Step 2: learn encoder
            label1 = future_states.mean(1, keepdims=True)       # N(0, 0.5)
            label2 = future_states.std(1, keepdims=True)        # range 1~2
            label2[label2 != label2] = 0
            if self.config.label_std_mapping:
                label2 = torch.logit((label2 / 3).clip(min=EPS, max=1 - EPS))
            labels = torch.cat((label1, label2), 1)
            if self.config.label_noise is not None:
                labels = labels \
                    + torch.normal(0, self.config.label_noise, size=labels.shape, device=labels.device)
            embedings = self.state_encoder(states)
            loss_encoder = nn.MSELoss()(embedings, labels)
            self.encoder_optim.zero_grad()
            loss_encoder.backward()
            self.encoder_optim.step()

            # Step 3: learn critic
            actions = self.actor(self.state_encoder(states), noise=self.config.noise).detach()
            rewards = self.get_reward(future_states, actions).detach()
            values = self.critic(self.state_encoder(states), self.action_encoder(actions))
            loss_critic = nn.MSELoss()(values, rewards)
            self.critic_optim.zero_grad()
            loss_critic.backward()
            self.critic_optim.step()

            # Step 4: learn actor
            actions, logits = self.actor(self.state_encoder(states), require_logits=True)
            loss_actor = - torch.mean(self.critic(self.state_encoder(states), self.action_encoder(actions))) \
                - self.config.entropy_coeff * Categorical(logits=logits).entropy().mean()
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            if i % 1000 == 0:
                e_eval = self.evaluate(actor_func)
                e_train = self.evaluate(actor_func, train=True)
                info.append(dict(
                    iterations=i,
                    loss_encoder=float(loss_encoder),
                    loss_actor=float(loss_actor),
                    loss_critic=float(loss_critic),
                    loss_total=float(loss_actor) + float(loss_critic) + float(loss_encoder),
                    reward_mean_train=e_train['reward_mean'],
                    reward_std_train=e_train['reward_std'],
                    reward_mean_eval=e_eval['reward_mean'],
                    reward_std_eval=e_eval['reward_std'],
                ))

            if i % 10000 == 0:
                pd.DataFrame(info).to_csv(os.path.join(self.config.result_path, 'result_tmp.csv'))
                filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}.png'.format(i))
                self.evaluate_sample(actor_func, filename)
                filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}_IS.png'.format(i))
                self.evaluate_sample_IS(actor_func, filename)
                filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}_SA.png'.format(i))
                self.evaluate_sample_sigma_alpha(actor_func, filename)

        return info


class AgentFitParam(Agent):
    def __init__(self, config):
        super(AgentFitParam, self).__init__(config)

        self.action_encoder = EncoderMLPNetwork(dim_input=self.dim_action, dim_output=self.config.action_emb_dim)\
            .to(device=self.config.device)
        self.state_encoder = EncoderMLPNetwork(dim_input=self.dim_state, dim_output=self.config.state_emb_dim)\
            .to(device=self.config.device)
        self.actor = ActorEmbMLPNetwork(dim_input=self.config.state_emb_dim, dim_output=self.dim_action)\
            .to(device=self.config.device)
        self.critic = CriticEmbMLPNetwork(dim_input1=self.config.state_emb_dim, dim_input2=self.config.action_emb_dim)\
            .to(device=self.config.device)

        critic_params = list(self.critic.parameters()) + list(self.action_encoder.parameters())
        self.actor_optim = opt.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optim = opt.Adam(critic_params, lr=self.config.critic_lr)
        self.encoder_optim = opt.Adam(self.state_encoder.parameters(), lr=self.config.encoder_lr)

        data, sigmas, alphas = self.generate_data(self.config.total_data)
        self.data = self._to_tensor(data)
        self.sigmas = self._to_tensor(sigmas)
        self.alphas = self._to_tensor(alphas)

    def learn(self):

        info = []
        actor_func = lambda states: self.actor(self.state_encoder(states))

        for i in trange(self.config.num_iterations):

            # Step 1: sample a batch
            inds = np.random.choice(self.config.total_data, self.config.batch_size)
            data = self.data[inds]
            sigmas = self.sigmas[inds]
            alphas = self.alphas[inds]

            states = data[:, :self.config.state_length]
            future_states = data[:, -self.config.future_length:]

            # Step 2: learn encoder
            label1 = states.mean(1, keepdims=True)       # N(0, 0.5)
            label2 = states.std(1, keepdims=True)        # range 1~2
            label2[label2 != label2] = 0
            if self.config.label_std_mapping:
                label2 = torch.logit((label2 / 3).clip(min=EPS, max=1 - EPS))
            labels = torch.cat((label1, label2), 1)
            if self.config.label_noise is not None:
                labels = labels \
                    + torch.normal(0, self.config.label_noise, size=labels.shape, device=labels.device)
            embedings = self.state_encoder(states)
            loss_encoder = nn.MSELoss()(embedings, labels)
            self.encoder_optim.zero_grad()
            loss_encoder.backward()
            self.encoder_optim.step()

            # Step 3: learn critic
            actions = self.actor(self.state_encoder(states), noise=self.config.noise).detach()
            rewards = self.get_reward(future_states, actions).detach()
            values = self.critic(self.state_encoder(states), self.action_encoder(actions))
            loss_critic = nn.MSELoss()(values, rewards)
            self.critic_optim.zero_grad()
            loss_critic.backward()
            self.critic_optim.step()

            # Step 4: learn actor
            actions, logits = self.actor(self.state_encoder(states), require_logits=True)
            loss_actor = - torch.mean(self.critic(self.state_encoder(states), self.action_encoder(actions))) \
                - self.config.entropy_coeff * Categorical(logits=logits).entropy().mean()
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            if i % 1000 == 0:
                e_eval = self.evaluate(actor_func)
                e_train = self.evaluate(actor_func, train=True)
                info.append(dict(
                    iterations=i,
                    loss_encoder=float(loss_encoder),
                    loss_actor=float(loss_actor),
                    loss_critic=float(loss_critic),
                    loss_total=float(loss_actor) + float(loss_critic) + float(loss_encoder),
                    reward_mean_train=e_train['reward_mean'],
                    reward_std_train=e_train['reward_std'],
                    reward_mean_eval=e_eval['reward_mean'],
                    reward_std_eval=e_eval['reward_std'],
                ))

            if i % 10000 == 0:
                pd.DataFrame(info).to_csv(os.path.join(self.config.result_path, 'result_tmp.csv'))
                filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}.png'.format(i))
                self.evaluate_sample(actor_func, filename)
                filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}_IS.png'.format(i))
                csv_filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}_IS.csv'.format(i))
                self.evaluate_sample_IS(actor_func, filename, csv_filename)
                filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}_SA.png'.format(i))
                self.evaluate_sample_sigma_alpha(actor_func, filename)

        return info


class AgentKnownParam(Agent):
    def __init__(self, config):
        super(AgentKnownParam, self).__init__(config)

        self.action_encoder = EncoderMLPNetwork(dim_input=self.dim_action, dim_output=self.config.action_emb_dim)\
            .to(device=self.config.device)
        self.state_encoder = EncoderMLPNetwork(dim_input=self.dim_state, dim_output=self.config.state_emb_dim)\
            .to(device=self.config.device)
        self.actor = ActorEmbMLPNetwork(dim_input=self.config.state_emb_dim, dim_output=self.dim_action)\
            .to(device=self.config.device)
        self.critic = CriticEmbMLPNetwork(dim_input1=self.config.state_emb_dim, dim_input2=self.config.action_emb_dim)\
            .to(device=self.config.device)

        critic_params = list(self.critic.parameters()) + list(self.action_encoder.parameters())
        self.actor_optim = opt.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optim = opt.Adam(critic_params, lr=self.config.critic_lr)
        self.encoder_optim = opt.Adam(self.state_encoder.parameters(), lr=self.config.encoder_lr)

        data, sigmas, alphas = self.generate_data(self.config.total_data)
        self.data = self._to_tensor(data)
        self.sigmas = self._to_tensor(sigmas)
        self.alphas = self._to_tensor(alphas)

    def learn(self):

        info = []
        actor_func = lambda states: self.actor(self.state_encoder(states))

        if self.config.pretrain:
            for i in trange(self.config.num_iterations // 10):
                # Step 1: sample a batch
                inds = np.random.choice(self.config.total_data, self.config.batch_size)
                data = self.data[inds]
                sigmas = self.sigmas[inds]
                alphas = self.alphas[inds]
                states = data[:, :self.config.state_length]

                # Step 2: learn encoder
                labels = torch.cat((sigmas, alphas), 1)
                embedings = self.state_encoder(states)
                loss_encoder = nn.MSELoss()(embedings, labels)
                self.encoder_optim.zero_grad()
                loss_encoder.backward()
                self.encoder_optim.step()

                if i % 1000 == 0:
                    info.append(dict(
                        iterations=i,
                        loss_encoder=float(loss_encoder),
                    ))

        for i in trange(self.config.num_iterations):

            # Step 1: sample a batch
            inds = np.random.choice(self.config.total_data, self.config.batch_size)
            data = self.data[inds]
            sigmas = self.sigmas[inds]
            alphas = self.alphas[inds]

            states = data[:, :self.config.state_length]
            future_states = data[:, -self.config.future_length:]

            if not self.config.pretrain:
                # Step 2: learn encoder
                labels = torch.cat((sigmas, alphas), 1)
                if self.config.label_noise is not None:
                    labels = labels \
                        + torch.normal(0, self.config.label_noise, size=labels.shape, device=labels.device)
                embedings = self.state_encoder(states)
                loss_encoder = nn.MSELoss()(embedings, labels)
                self.encoder_optim.zero_grad()
                loss_encoder.backward()
                self.encoder_optim.step()

            # Step 3: learn critic
            actions = self.actor(self.state_encoder(states), noise=self.config.noise).detach()
            rewards = self.get_reward(future_states, actions).detach()
            values = self.critic(self.state_encoder(states), self.action_encoder(actions))
            loss_critic = nn.MSELoss()(values, rewards)
            self.critic_optim.zero_grad()
            loss_critic.backward()
            self.critic_optim.step()

            # Step 4: learn actor
            actions, logits = self.actor(self.state_encoder(states), require_logits=True)
            loss_actor = - torch.mean(self.critic(self.state_encoder(states), self.action_encoder(actions))) \
                - self.config.entropy_coeff * Categorical(logits=logits).entropy().mean()
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            if i % 1000 == 0:
                e_eval = self.evaluate(actor_func)
                e_train = self.evaluate(actor_func, train=True)
                info.append(dict(
                    iterations=i,
                    loss_encoder=float(loss_encoder),
                    loss_actor=float(loss_actor),
                    loss_critic=float(loss_critic),
                    loss_total=float(loss_actor) + float(loss_critic) + float(loss_encoder),
                    reward_mean_train=e_train['reward_mean'],
                    reward_std_train=e_train['reward_std'],
                    reward_mean_eval=e_eval['reward_mean'],
                    reward_std_eval=e_eval['reward_std'],
                ))

            if i % 10000 == 0:
                pd.DataFrame(info).to_csv(os.path.join(self.config.result_path, 'result_tmp.csv'))
                filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}.png'.format(i))
                self.evaluate_sample(actor_func, filename)

        return info


class AgentRLBottleneck(Agent):
    def __init__(self, config):
        super(AgentRLBottleneck, self).__init__(config)

        if config.bottleneck_mode == 'separate':
            self.action_encoder = EncoderMLPNetwork(dim_input=self.dim_action, dim_output=self.config.action_emb_dim)\
                .to(device=self.config.device)
            self.state_encoder1 = EncoderMLPNetwork(dim_input=self.dim_state, dim_output=self.config.state_emb_dim)\
                .to(device=self.config.device)
            self.state_encoder2 = EncoderMLPNetwork(dim_input=self.dim_state, dim_output=self.config.state_emb_dim)\
                .to(device=self.config.device)
            self.actor = ActorEmbMLPNetwork(dim_input=self.config.state_emb_dim, dim_output=self.dim_action)\
                .to(device=self.config.device)
            self.critic = CriticEmbMLPNetwork(dim_input1=self.config.state_emb_dim, dim_input2=self.config.action_emb_dim)\
                .to(device=self.config.device)

            actor_params = list(self.actor.parameters()) + list(self.state_encoder1.parameters())
            critic_params = list(self.critic.parameters()) + list(self.state_encoder2.parameters()) \
                + list(self.action_encoder.parameters())
            self.actor_optim = opt.Adam(actor_params, lr=self.config.actor_lr)
            self.critic_optim = opt.Adam(critic_params, lr=self.config.critic_lr)

            self.actor_func = lambda states, noise=None, require_logits=False: \
                self.actor(self.state_encoder1(states), noise=noise, require_logits=require_logits)
            self.critic_func = lambda states, actions: \
                self.critic(self.state_encoder2(states), self.action_encoder(actions))
        elif config.bottleneck_mode == 'shared':
            self.action_encoder = EncoderMLPNetwork(dim_input=self.dim_action, dim_output=self.config.action_emb_dim)\
                .to(device=self.config.device)
            self.state_encoder = EncoderMLPNetwork(dim_input=self.dim_state, dim_output=self.config.state_emb_dim)\
                .to(device=self.config.device)
            self.actor = ActorEmbMLPNetwork(dim_input=self.config.state_emb_dim, dim_output=self.dim_action)\
                .to(device=self.config.device)
            self.critic = CriticEmbMLPNetwork(dim_input1=self.config.state_emb_dim, dim_input2=self.config.action_emb_dim)\
                .to(device=self.config.device)

            actor_params = list(self.actor.parameters()) + list(self.state_encoder.parameters())
            critic_params = list(self.critic.parameters()) + list(self.state_encoder.parameters()) \
                + list(self.action_encoder.parameters())
            self.actor_optim = opt.Adam(actor_params, lr=self.config.actor_lr)
            self.critic_optim = opt.Adam(critic_params, lr=self.config.critic_lr)

            self.actor_func = lambda states, noise=None, require_logits=False: \
                self.actor(self.state_encoder(states), noise=noise, require_logits=require_logits)
            self.critic_func = lambda states, actions: \
                self.critic(self.state_encoder(states), self.action_encoder(actions))
        elif config.bottleneck_mode == 'shared_separate_opt':
            self.action_encoder = EncoderMLPNetwork(dim_input=self.dim_action, dim_output=self.config.action_emb_dim)\
                .to(device=self.config.device)
            self.state_encoder = EncoderMLPNetwork(dim_input=self.dim_state, dim_output=self.config.state_emb_dim)\
                .to(device=self.config.device)
            self.actor = ActorEmbMLPNetwork(dim_input=self.config.state_emb_dim, dim_output=self.dim_action)\
                .to(device=self.config.device)
            self.critic = CriticEmbMLPNetwork(dim_input1=self.config.state_emb_dim, dim_input2=self.config.action_emb_dim)\
                .to(device=self.config.device)

            actor_params = list(self.actor.parameters()) + list(self.state_encoder.parameters())
            critic_params = list(self.critic.parameters()) + list(self.action_encoder.parameters())
            self.actor_optim = opt.Adam(actor_params, lr=self.config.actor_lr)
            self.critic_optim = opt.Adam(critic_params, lr=self.config.critic_lr)

            self.actor_func = lambda states, noise=None, require_logits=False: \
                self.actor(self.state_encoder(states), noise=noise, require_logits=require_logits)
            self.critic_func = lambda states, actions: \
                self.critic(self.state_encoder(states), self.action_encoder(actions))


        data, sigmas, alphas = self.generate_data(self.config.total_data)
        self.data = self._to_tensor(data)
        self.sigmas = self._to_tensor(sigmas)
        self.alphas = self._to_tensor(alphas)

    def learn(self):

        info = []

        for i in trange(self.config.num_iterations):

            # Step 1: sample a batch
            inds = np.random.choice(self.config.total_data, self.config.batch_size)
            data = self.data[inds]
            sigmas = self.sigmas[inds]
            alphas = self.alphas[inds]

            states = data[:, :self.config.state_length]
            future_states = data[:, -self.config.future_length:]

            # Step 2: learn critic
            actions = self.actor_func(states, noise=self.config.noise).detach()
            rewards = self.get_reward(future_states, actions).detach()
            values = self.critic_func(states, actions)
            loss_critic = nn.MSELoss()(values, rewards)
            self.critic_optim.zero_grad()
            loss_critic.backward()
            self.critic_optim.step()

            # Step 3: learn actor
            actions, logits = self.actor_func(states, require_logits=True)
            loss_actor = - torch.mean(self.critic_func(states, actions)) \
                - self.config.entropy_coeff * Categorical(logits=logits).entropy().mean()
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            if i % 1000 == 0:
                e_eval = self.evaluate(self.actor_func)
                e_train = self.evaluate(self.actor_func, train=True)
                info.append(dict(
                    loss_actor=float(loss_actor),
                    loss_critic=float(loss_critic),
                    loss_total=float(loss_actor) + float(loss_critic),
                    iterations=i,
                    reward_mean_train=e_train['reward_mean'],
                    reward_std_train=e_train['reward_std'],
                    reward_mean_eval=e_eval['reward_mean'],
                    reward_std_eval=e_eval['reward_std'],
                ))

            if i % 10000 == 0:
                pd.DataFrame(info).to_csv(os.path.join(self.config.result_path, 'result_tmp.csv'))
                filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}.png'.format(i))
                self.evaluate_sample(self.actor_func, filename)

        return info


class AgentRL(Agent):
    def __init__(self, config):
        super(AgentRL, self).__init__(config)
        self.actor = ActorMLPNetwork(dim_input=self.dim_state, dim_output=self.dim_action).to(device=self.config.device)
        self.critic = CriticMLPNetwork(dim_input=self.config.total_length, dim_output=1).to(device=self.config.device)
        self.actor_optim = opt.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optim = opt.Adam(self.critic.parameters(), lr=self.config.critic_lr)

        data, sigmas, alphas = self.generate_data(self.config.total_data)
        self.data = self._to_tensor(data)
        self.sigmas = self._to_tensor(sigmas)
        self.alphas = self._to_tensor(alphas)

    def learn(self):

        info = []
        for i in trange(self.config.num_iterations):

            # Step 1: sample a batch
            inds = np.random.choice(self.config.total_data, self.config.batch_size)
            data = self.data[inds]
            sigmas = self.sigmas[inds]
            alphas = self.alphas[inds]

            states = data[:, :self.config.state_length]
            future_states = data[:, -self.config.future_length:]

            # Step 2: learn critic
            actions = self.actor(states, noise=self.config.noise).detach()
            rewards = self.get_reward(future_states, actions).detach()
            values = self.critic(states, actions)
            loss_critic = nn.MSELoss()(values, rewards)
            self.critic_optim.zero_grad()
            loss_critic.backward()
            self.critic_optim.step()

            # Step 3: learn actor
            actions, logits = self.actor(states, require_logits=True)
            loss_actor = - torch.mean(self.critic(states, actions)) \
                - self.config.entropy_coeff * Categorical(logits=logits).entropy().mean()
            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            if i % 1000 == 0:
                e_eval = self.evaluate(self.actor)
                e_train = self.evaluate(self.actor, train=True)
                info.append(dict(
                    loss_actor=float(loss_actor),
                    loss_critic=float(loss_critic),
                    loss_total=float(loss_actor) + float(loss_critic),
                    iterations=i,
                    reward_mean_train=e_train['reward_mean'],
                    reward_std_train=e_train['reward_std'],
                    reward_mean_eval=e_eval['reward_mean'],
                    reward_std_eval=e_eval['reward_std'],
                ))

            if i % 10000 == 0:
                pd.DataFrame(info).to_csv(os.path.join(self.config.result_path, 'result_tmp.csv'))
                filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}.png'.format(i))
                self.evaluate_sample(self.actor, filename)
                filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}_IS.png'.format(i))
                csv_filename = os.path.join(self.config.result_path, 'evaluate', 'it_{:09d}_IS.csv'.format(i))
                self.evaluate_sample_IS(self.actor, filename, csv_filename)

        return info


def run_exp32(argus):

    algo, total_data, parallel_id = argus

    config = DefaultConfig()
    config.total_data = total_data
    config.label_noise = 5.0 if 'noise' in algo else None
    id_str = 'v1_{}_data{}k_{}'.format(algo, total_data // 1000, parallel_id)
    config.result_path = os.path.join(config.result_path, id_str)
    os.makedirs(config.result_path, exist_ok=True)  
    extend_path = lambda x: os.path.join(config.result_path, x)
    os.makedirs(extend_path('evaluate'), exist_ok=True)

    if algo == 'predict_param' or algo == 'predict_param_noise':
        agent = AgentPredictParam(config)
    elif algo == 'predict_full':
        agent = AgentPredictFull(config)
    elif algo == 'known_param' or algo == 'known_param_noise':
        agent = AgentKnownParam(config)
    elif algo == 'fit_param' or algo == 'fit_param_noise':
        agent = AgentFitParam(config)
    elif algo == 'RL':
        agent = AgentRL(config)
    elif algo == 'RL_bottleneck':
        agent = AgentRLBottleneck(config)

    info = agent.learn()
    info = pd.DataFrame(info)
    info.to_csv(extend_path('result.csv'))

    total_length = info.shape[0]
    last_length = total_length // 10
    reward_mean_train = np.mean(info['reward_mean_train'].values[-last_length:])
    reward_mean_eval = np.mean(info['reward_mean_eval'].values[-last_length:])
    return dict(
        algo=algo,
        total_data='{}k'.format(total_data // 1000),
        parallel_id=parallel_id,
        reward_mean_train=reward_mean_train,
        reward_mean_eval=reward_mean_eval,
        reward_std_train=np.mean(info['reward_std_train'].values[-last_length:]),
        reward_std_eval=np.mean(info['reward_std_eval'].values[-last_length:]),
        gap=reward_mean_train - reward_mean_eval
    )


if __name__ == '__main__':


    algo_list = [
        'predict_param',
        'predict_param_noise',
        'predict_full',
        'known_param',
        'known_param_noise',
        'fit_param',
        'fit_param_noise',
        'RL',
        'RL_bottleneck',
    ]

    test_list = list(itertools.product(algo_list, [100000, 10000, 1000], np.arange(5)))
    pool = Pool(6)
    record = pool.map(run_exp32, test_list)
    record = pd.DataFrame(record)
    record.to_csv(os.path.join(DefaultConfig().result_path, 'result_original.csv'))
    stats = record.groupby(['total_data', 'algo']).agg([np.mean, np.std])
    stats.to_csv(os.path.join(DefaultConfig().result_path, 'result_stats.csv'))
