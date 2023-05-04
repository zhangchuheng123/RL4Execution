"""
Algo1: Use hand-crafted features to learn a compact context representation
"""

import torch
import torch.nn as nn
import torch.optim as opt
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
    
from constants import CODE_LIST, JUNE_DATE_LIST, VALIDATION_DATE_LIST, VALIDATION_CODE_LIST
from env import make_env

from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax, expit
from collections import deque
from tqdm import trange
import pandas as pd
import numpy as np
import itertools
import pdb
import os 


FEATURE_SET_LOB = [
    'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
    'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',  
    'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5', 
    'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5', 
    'high_low_price_diff', 'close_price', 'volume', 'vwap', 'time_diff',
    'ask_bid_spread', 'immediate_market_order_cost_bid'
]

FEATURE_SET_FULL = FEATURE_SET_LOB + [
    'ab_volume_misbalance', 'transaction_net_volume', 
    'volatility', 'trend', 'weighted_price', 'order_imblance',
    'VOLR', 'PCTN_1min', 'MidMove_1min', 'trend_strength'
]


class DefaultConfig(object):

    path_raw_data = '/data/execution_data_v2/raw'
    # path_pkl_data = '/data/execution_data/pkl'
    path_pkl_data = '/mnt/execution_data_v2/pkl'
    result_path = 'results/exp46'

    code_list = CODE_LIST
    date_list = JUNE_DATE_LIST
    code_list_validation = VALIDATION_CODE_LIST
    date_list_validation = VALIDATION_DATE_LIST

    # ############################### Special Parameters for Our Algorithm ###############################
    # Encoder learning
    agent_encoder_num_iterations = 2000
    agent_encoder_num_trajs = 100
    agent_encoder_batch_size = 128
    agent_encoder_num_epochs = (100 * 20 * 4) // 128
    agent_policy_selected_features = [
        'vwap', 'close_price', 'askPrice1', 'bidPrice1', 'askVolume1', 'bidVolume1',
        'time_diff', 'ask_bid_spread', 'immediate_market_order_cost_bid'
    ]
    # ############################### END ###############################

    agent_scale = 100000
    agent_batch_size = 128
    agent_learn_start = 5000
    agent_gamma = 0.998
    agent_epsilon = 0.7
    agent_total_steps = 20 * agent_scale
    agent_buffer_size = agent_scale
    agent_network_update_freq = 4
    # Smooth L1 loss (SL1) or mean squared error (MSE)
    agent_loss_type = 'SL1'
    agent_learning_rate = 1e-3
    agent_lr_decay_freq = 2000
    agent_target_update_freq = 2000
    agent_eval_freq = 2000
    # Becomes 0.01 upon 70% of the training
    agent_epsilon_decay = np.exp(np.log(0.01) / (agent_scale * 0.5))
    agent_plot_freq = 20000
    agent_device = 'cuda'

    # ############################### Trade Setting Parameters ###############################
    # Planning horizon is 30mins
    simulation_planning_horizon = 30
    # Total volume to trade w.r.t. the basis volume
    simulation_volume_ratio = 0.005
    # Order volume = total volume / simulation_num_shares
    simulation_num_shares = 10
    # Maximum quantity is total_quantity / simulation_num_shares; further devide this into 3 levels
    simulation_discrete_quantities = 3
    # Choose the wrapper
    simulation_action_type = 'discrete_pq'
    # Discrete action space
    simulation_discrete_actions = \
        list(itertools.product(
            np.concatenate([[-50, -40, -30, -25, -20, -15], np.linspace(-10, 10, 21), [15, 20, 25, 30, 40, 50]]),
            np.arange(simulation_discrete_quantities) + 1
        ))
    # Stack the features of the previous x bars
    simulation_loockback_horizon = 5
    # Whether return flattened or stacked features of the past x bars
    simulation_do_feature_flatten = True
    simulation_direction = 'sell'
    # If the quantity is not fully filled at the last time step, we place an MO to liquidate and further plus a penalty
    simulation_not_filled_penalty_bp = 2.0
    # Scale the price delta if we use continuous actions (deprecated)
    simulation_continuous_action_scale = 10
    # ############################### END ###############################

    # ############################### Sweep Parameters ###############################
    # Encourage a uniform liquidation strategy
    simulation_linear_reg_coeff = [1.0, 0.1, 0.01]
    agent_learning_rate = [1e-4, 1e-5]
    agent_network_structrue = 'MLPNetwork'
    simulation_features = FEATURE_SET_LOB
    # ############################### END ###############################

    # For interprebility logging
    evaluation_embedding_record_size = 3000


class MLPNetwork(nn.Module):
    def __init__(self, dim_input1, dim_input2, dim_embedding, dim_output, market_state3_indices, hidden=128):
        super(MLPNetwork, self).__init__()
        
        self.dim_input1 = dim_input1
        self.dim_input2 = dim_input2
        self.dim_input3 = len(market_state3_indices)
        self.dim_embedding = dim_embedding
        self.dim_output = dim_output
        self.market_state3_indices = market_state3_indices
        
        self.fc1 = nn.Linear(dim_input1, 2 * hidden)
        self.fc2 = nn.Linear(2 * hidden, hidden)
        self.fc3 = nn.Linear(dim_input2, hidden)
        self.fc4 = nn.Linear(2 * hidden, dim_embedding)

        self.fc51 = nn.Linear(dim_embedding, hidden)
        self.fc52 = nn.Linear(dim_input2, hidden)
        self.fc53 = nn.Linear(self.dim_input3, hidden)
        self.fc6 = nn.Linear(3 * hidden, hidden)
        self.fc7 = nn.Linear(hidden, dim_output)

    def encoding(self, market_states, private_states):
        x = F.relu(self.fc1(market_states))
        x = F.relu(self.fc2(x))
        y = F.relu(self.fc3(private_states))
        z = torch.cat((x, y), 1)
        z = self.fc4(z)
        return z
        
    def forward(self, market_states, private_states, emb=None):
        if emb is None:
            emb = self.encoding(market_states, private_states).detach()
        x = F.relu(self.fc51(emb))
        y = F.relu(self.fc52(private_states))
        market_states3 = market_states[:, self.market_state3_indices]
        z = F.relu(self.fc53(market_states3))
        out = torch.cat((x, y, z), 1)
        out = F.relu(self.fc6(out))
        out = self.fc7(out)
        return out

    def act(self, market_state, private_state, device='cuda'):
        market_state = Tensor(market_state).unsqueeze(0).to(device=device)
        private_state = Tensor(private_state).unsqueeze(0).to(device=device)
        return int(self.forward(market_state, private_state).argmax(1)[0])
        
    def act_egreedy(self, market_state, private_state, e=0.7, device='cuda'):
        return self.act(market_state, private_state, device=device) if np.random.rand() > e \
            else np.random.randint(self.dim_output)


class MLPNetworkLarge(nn.Module):
    def __init__(self, dim_input1, dim_input2, dim_embedding, dim_output, market_state3_indices, hidden=256):
        super(MLPNetworkLarge, self).__init__()
        
        self.dim_input1 = dim_input1
        self.dim_input2 = dim_input2
        self.dim_input3 = len(market_state3_indices)
        self.dim_embedding = dim_embedding
        self.dim_output = dim_output
        self.market_state3_indices = market_state3_indices
        
        # e: encoder p: policy m: market p: private e: embedding
        self.fc_em1 = nn.Linear(dim_input1, hidden)
        self.fc_em2 = nn.Linear(hidden, hidden)
        self.fc_ep1 = nn.Linear(dim_input2, hidden)
        self.fc_e3 = nn.Linear(2 * hidden, hidden)
        self.fc_e4 = nn.Linear(hidden, dim_embedding)

        self.fc_pe1 = nn.Linear(dim_embedding, hidden)
        self.fc_pp1 = nn.Linear(dim_input2, hidden)
        self.fc_pm1 = nn.Linear(self.dim_input3, hidden)
        self.fc_p2 = nn.Linear(3 * hidden, hidden)
        self.fc_p3 = nn.Linear(hidden, hidden)
        self.fc_p4 = nn.Linear(hidden, dim_output)

    def encoding(self, market_states, private_states):
        x = F.relu(self.fc_em1(market_states))
        x = F.relu(self.fc_em2(x))
        y = F.relu(self.fc_ep1(private_states))
        z = torch.cat((x, y), 1)
        z = F.relu(self.fc_e3(z))
        z = self.fc_e4(z)
        return z
        
    def forward(self, market_states, private_states, emb=None):
        if emb is None:
            emb = self.encoding(market_states, private_states).detach()
        x = F.relu(self.fc_pe1(emb))
        y = F.relu(self.fc_pp1(private_states))
        market_states3 = market_states[:, self.market_state3_indices]
        z = F.relu(self.fc_pm1(market_states3))
        out = torch.cat((x, y, z), 1)
        out = F.relu(self.fc_p2(out))
        out = F.relu(self.fc_p3(out))
        out = self.fc_p4(out)
        return out

    def act(self, market_state, private_state, device='cuda'):
        market_state = Tensor(market_state).unsqueeze(0).to(device=device)
        private_state = Tensor(private_state).unsqueeze(0).to(device=device)
        return int(self.forward(market_state, private_state).argmax(1)[0])
        
    def act_egreedy(self, market_state, private_state, e=0.7, device='cuda'):
        return self.act(market_state, private_state, device=device) if np.random.rand() > e \
            else np.random.randint(self.dim_output)


class ReplayBuffer(object):
    """docstring for ReplayBuffer"""
    def __init__(self, maxlen):
        super(ReplayBuffer, self).__init__()
        self.maxlen = maxlen
        self.data = deque(maxlen=maxlen)
        
    def push(self, *args):
        self.data.append(args)

    def sample(self, batch_size):
        inds = np.random.choice(len(self.data), batch_size, replace=False)
        return zip(*[self.data[i] for i in inds])

    def sample_all(self):
        return zip(*list(self.data))

    def update_all(self, new_data, ind):
        for i in range(len(self.data)):
            tup = list(self.data[i])
            tup[ind] = new_data[i, :]
            self.data[i] = tuple(tup)


class Agent(object):
    def __init__(self, config, pid=0):
        super(Agent, self).__init__()

        self._set_seed()

        if config.agent_device == 'cuda':
            total_cudas = torch.cuda.device_count()
            self.device = 'cuda:{}'.format(pid % total_cudas)
        else:
            self.device = config.agent_device

        self.config = config
        self.env = make_env(config)
        self.dim_input1 = self.env.observation_dim       # dimension of market states
        self.dim_input2 = 2                              # dimension of private states
        self.dim_input3 = len(self.config.agent_policy_selected_features)
        self.dim_output = self.env.action_dim
        self.market_state3_indices = \
            [config.simulation_features.index(item) for item in config.agent_policy_selected_features]
        network = config.agent_network_structrue
        self.network = network(self.dim_input1, self.dim_input2, 8, self.dim_output, self.market_state3_indices)\
            .to(device=self.device)
        self.network_target = network(self.dim_input1, self.dim_input2, 8, self.dim_output, self.market_state3_indices)\
            .to(device=self.device)
        self.network_target.load_state_dict(self.network.state_dict())
        self.optimizer = opt.Adam(self.network.parameters(), lr=config.agent_learning_rate)
        self.scheduler = opt.lr_scheduler.StepLR(self.optimizer, step_size=config.agent_lr_decay_freq, gamma=0.998)
        self.buffer = ReplayBuffer(self.config.agent_buffer_size)
        self.evaluation = Evaluation(self.config)
        if config.agent_loss_type == 'MSE':
            self.loss_func = nn.MSELoss()
        elif config.agent_loss_type == 'SL1':
            self.loss_func = F.smooth_l1_loss
        self.ms_scaler = StandardScaler()

        self.record_embeddings = None
        self.record_marketsates = None

    def _reset_optimizer(self):
        self.optimizer = opt.Adam(self.network.parameters(), lr=self.config.agent_learning_rate)
        self.scheduler = opt.lr_scheduler.StepLR(self.optimizer, step_size=self.config.agent_lr_decay_freq, gamma=0.998)

    def _set_seed(self, seed=None):
        if seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder='little')
        else:
            seed = seed + 1234
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _filter(state):
        return np.clip(state, -3, 3)

    def _to_tensor(self, tensor, dtype=torch.float):
        return torch.tensor(tensor, dtype=dtype, device=self.device)

    def _clip_grad(self, val=1):
        for param in self.network.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-val, val)

    def _get_label(self):
        labels = self.env.get_future(['vwap', 'ask_bid_spread'])
        labels = np.array([
            labels['vwap'].max() - labels['vwap'].values[0],
            labels['vwap'].min() - labels['vwap'].values[0],
            labels['vwap'].mean() - labels['vwap'].values[0],
            labels['vwap'].std(),
            labels['ask_bid_spread'].max() - labels['ask_bid_spread'].values[0],
            labels['ask_bid_spread'].min() - labels['ask_bid_spread'].values[0],
            labels['ask_bid_spread'].mean() - labels['ask_bid_spread'].values[0],
            labels['ask_bid_spread'].std(),
        ])
        return labels

    def learn_simutaneous(self):

        train_record = []
        eval_record = []
        reward = 0
        eplen = 0
        loss = 0
        loss_encoder = 0
        loss_policy = 0
        avg_Q = 0
        epsilon = self.config.agent_epsilon

        sm, sp = self.env.reset()
        label = self._get_label()

        for i in trange(self.config.agent_total_steps):

            # Step 1: Execute one step and store it to the replay buffer
            if i <= self.config.agent_learn_start:
                a = self.env.action_sample_func()
            else:
                tsm = self.ms_scaler.transform(sm.reshape(1, -1)).flatten()
                a = self.network.act_egreedy(tsm, sp, e=epsilon, device=self.device)

            nsm, nsp, r, done, info = self.env.step(a)
            self.buffer.push(sm, sp, a, r, nsm, nsp, done, label)
            reward += r
            eplen += 1
            if done:
                train_record.append(dict(
                    i=i, 
                    reward=reward,
                    eplen=eplen,
                    epsilon=epsilon,
                    lr=self.optimizer.param_groups[0]['lr'],
                    loss_encoder=float(loss_encoder),
                    loss_policy=float(loss_policy),
                    loss=float(loss),
                    avg_Q=float(avg_Q),
                    BP=self.env.get_metric('BP'),
                    IS=self.env.get_metric('IS'),
                    code=info['code'],
                    date=info['date'],
                    start_index=info['start_index']
                    ))
                reward = 0
                eplen = 0
                epsilon = max(0.01, epsilon * self.config.agent_epsilon_decay)
                sm, sp = self.env.reset()
            else:
                sm, sp = nsm, nsp
            label = self._get_label()

            # Step 2: Estimate variance for market states
            if i == self.config.agent_learn_start:
                market_states, _, _, _, nmarket_states, _, _, _ = self.buffer.sample_all()
                self.ms_scaler.fit(np.array(market_states))

                # Since we will use the buffer later, so we need to scale the market states in the buffer
                self.buffer.update_all(self.ms_scaler.transform(market_states), 0)
                self.buffer.update_all(self.ms_scaler.transform(nmarket_states), 4)
            
            # Step 4: Update the network every several steps
            if i >= self.config.agent_learn_start and i % self.config.agent_network_update_freq == 0:
                
                # sample a batch from the replay buffer
                bsm, bsp, ba, br, bnsm, bnsp, bd, blabel = self.buffer.sample(self.config.agent_batch_size)

                market_states = self._to_tensor(self._filter(self.ms_scaler.transform(np.array(bsm))))
                private_states = self._to_tensor(np.array(bsp))
                actions = self._to_tensor(np.array(ba), dtype=torch.long)
                rewards = self._to_tensor(np.array(br))
                nmarket_states = self._to_tensor(self._filter(self.ms_scaler.transform(np.array(bnsm))))
                nprivate_states = self._to_tensor(np.array(bnsp))
                masks = self._to_tensor(1 - np.array(bd) * 1)
                nactions = self.network(nmarket_states, nprivate_states).argmax(1)
                labels = self._to_tensor(np.array(blabel))

                # Update the encoder
                pred = self.network.encoding(market_states, private_states)
                loss_encoder = self.loss_func(pred, labels)

                # Update the policy
                Qtarget = (rewards + masks * self.config.agent_gamma * \
                    self.network_target(nmarket_states, nprivate_states)\
                    [range(self.config.agent_batch_size), nactions]).detach()
                Qvalue = self.network(market_states, private_states)\
                    [range(self.config.agent_batch_size), actions]
                avg_Q = Qvalue.mean().detach()
                loss_policy = self.loss_func(Qvalue, Qtarget)

                self.network.zero_grad()
                loss = loss_encoder * 4 + loss_policy
                loss.backward()
                self._clip_grad()
                self.optimizer.step()
                self.scheduler.step()
                
            # Step 5: Update target network
            if i % self.config.agent_target_update_freq == 0:
                self.network_target.load_state_dict(self.network.state_dict())

            # Step 6: Evaluate and log performance
            if i % self.config.agent_plot_freq == 0 and len(train_record) > 0:
                eval_agent = self._eval_agent(i)
                self.evaluation.evaluate_detail_batch(eval_agent, iteration=i)
                print(train_record[-1])

            if i % self.config.agent_eval_freq == 0:
                eval_agent = self._eval_agent(i)
                eval_record.append(self.evaluation.evaluate(eval_agent))
                print(eval_record[-1])

        return train_record, eval_record

    def learn_separate(self):

        self._estimate_scaler()
        encoder_record = self._learn_encoder()
        self.network_target.load_state_dict(self.network.state_dict())
        train_record, eval_record = self._learn_policy()
        return encoder_record, train_record, eval_record

    def _estimate_scaler(self):
        sm, sp = self.env.reset()

        for i in trange(self.config.agent_learn_start):
            a = self.env.action_sample_func()
            nsm, nsp, r, done, info = self.env.step(a)
            self.buffer.push(sm, sp, a, r, nsm, nsp, done)
            if done:
                sm, sp = self.env.reset()
            else:
                sm, sp = nsm, nsp

        market_states, _, _, _, nmarket_states, _, _ = self.buffer.sample_all()
        self.ms_scaler.fit(np.array(market_states))

        # Since we will use the buffer later, so we need to scale the market states in the buffer
        self.buffer.update_all(self.ms_scaler.transform(market_states), 0)
        self.buffer.update_all(self.ms_scaler.transform(nmarket_states), 4)

    def _learn_encoder(self):

        vwap_index = self.config.simulation_features.index('vwap')
        sprd_index = self.config.simulation_features.index('ask_bid_spread')
        indices = [vwap_index, sprd_index]
        record = []

        for i in trange(self.config.agent_encoder_num_iterations):

            # Step 1: Sample
            market_states = []
            private_states = []
            targets = []
            for j in range(self.config.agent_encoder_num_trajs):
                sm, sp = self.env.reset()
                features = []
                done = False
                while not done:
                    sm = self.ms_scaler.transform(sm.reshape(1, -1)).flatten()
                    market_states.append(sm)
                    private_states.append(sp)
                    features.append(sm[indices])
                    sm, sp, _, done, info = self.env.step(0)

                labels = pd.DataFrame(features[::-1], columns=['vwap', 'sprd'])

                label1 = labels['vwap'].rolling(len(features), min_periods=0).max() - labels['vwap']
                label2 = labels['vwap'].rolling(len(features), min_periods=0).min() - labels['vwap']
                label3 = labels['vwap'].rolling(len(features), min_periods=0).mean() - labels['vwap']
                label4 = labels['vwap'].rolling(len(features), min_periods=0).std().fillna(0)
                label5 = labels['sprd'].rolling(len(features), min_periods=0).max() - labels['sprd']
                label6 = labels['sprd'].rolling(len(features), min_periods=0).min() - labels['sprd']
                label7 = labels['sprd'].rolling(len(features), min_periods=0).mean() - labels['sprd']
                label8 = labels['sprd'].rolling(len(features), min_periods=0).std().fillna(0)

                labels = pd.concat([label1, label2, label3, label4, label5, label6, label7, label8], axis=1)[::-1].values
                # delete the last one
                targets.append(labels)

            market_states = np.stack(market_states, axis=0)
            private_states = np.stack(private_states, axis=0)
            targets = np.concatenate(targets, axis=0)

            # Step 2: Train encoder
            loss_mean = []
            for j in range(self.config.agent_encoder_num_epochs):
                inds = np.random.choice(len(market_states), self.config.agent_encoder_batch_size)
                ms = self._to_tensor(self._filter(market_states[inds, :]))
                ps = self._to_tensor(private_states[inds, :])
                tg = self._to_tensor(targets[inds, :])
                pred = self.network.encoding(ms, ps)

                loss = self.loss_func(pred, tg)
                self.network.zero_grad()
                loss.backward()
                self._clip_grad()
                self.optimizer.step()
                self.scheduler.step()
                loss_mean.append(float(loss))

            record.append(dict(loss=np.mean(loss_mean)))

        self._reset_optimizer()

        # Step additional: Record average embedding
        ms = self._to_tensor(self._filter(market_states))
        ps = self._to_tensor(private_states)
        self.record_embeddings = self.network.encoding(ms, ps).detach()
        self.record_marketsates = ms

        return record

    def _learn_policy(self):

        train_record = []
        eval_record = []
        reward = 0
        eplen = 0
        loss = 0
        avg_Q = 0
        epsilon = self.config.agent_epsilon

        sm, sp = self.env.reset()

        for i in trange(self.config.agent_total_steps):

            # Step 1: Execute one step and store it to the replay buffer
            tsm = self.ms_scaler.transform(sm.reshape(1, -1)).flatten()
            a = self.network.act_egreedy(tsm, sp, e=epsilon, device=self.device)
            nsm, nsp, r, done, info = self.env.step(a)
            self.buffer.push(sm, sp, a, r, nsm, nsp, done)
            reward += r
            eplen += 1
            if done:
                train_record.append(dict(
                    i=i, 
                    reward=reward,
                    eplen=eplen,
                    epsilon=epsilon,
                    lr=self.optimizer.param_groups[0]['lr'],
                    loss=float(loss),
                    avg_Q=float(avg_Q),
                    BP=self.env.get_metric('BP'),
                    IS=self.env.get_metric('IS'),
                    code=info['code'],
                    date=info['date'],
                    start_index=info['start_index']
                    ))
                reward = 0
                eplen = 0
                epsilon = max(0.01, epsilon * self.config.agent_epsilon_decay)
                sm, sp = self.env.reset()
            else:
                sm, sp = nsm, nsp
            
            # Step 2: Update the network every several steps
            if i % self.config.agent_network_update_freq == 0:
                
                # sample a batch from the replay buffer
                bsm, bsp, ba, br, bnsm, bnsp, bd = self.buffer.sample(self.config.agent_batch_size)

                market_states = self._to_tensor(self._filter(self.ms_scaler.transform(np.array(bsm))))
                private_states = self._to_tensor(np.array(bsp))
                actions = self._to_tensor(np.array(ba), dtype=torch.long)
                rewards = self._to_tensor(np.array(br))
                nmarket_states = self._to_tensor(self._filter(self.ms_scaler.transform(np.array(bnsm))))
                nprivate_states = self._to_tensor(np.array(bnsp))
                masks = self._to_tensor(1 - np.array(bd) * 1)
                nactions = self.network(nmarket_states, nprivate_states).argmax(1)

                Qtarget = (rewards + masks * self.config.agent_gamma * \
                    self.network_target(nmarket_states, nprivate_states)\
                    [range(self.config.agent_batch_size), nactions]).detach()
                Qvalue = self.network(market_states, private_states)\
                    [range(self.config.agent_batch_size), actions]
                avg_Q = Qvalue.mean().detach()
                loss = self.loss_func(Qvalue, Qtarget)
                self.network.zero_grad()
                loss.backward()
                self._clip_grad()
                # print('Finish the {}-th iteration, the loss = {}'.format(i, float(loss)))
                self.optimizer.step()
                self.scheduler.step()
                
            # Step 3: Update target network
            if i % self.config.agent_target_update_freq == 0:
                self.network_target.load_state_dict(self.network.state_dict())

            # Step 4: Evaluate and log performance
            # if i % self.config.agent_plot_freq == 0 and len(train_record) > 0:
            if i % self.config.agent_plot_freq == 0:
                eval_agent = self._eval_agent(i)
                self.evaluation.evaluate_detail_batch(eval_agent, iteration=i)

            if i % self.config.agent_eval_freq == 0:
                eval_agent = self._eval_agent(i)
                eval_record.append(self.evaluation.evaluate(eval_agent))
                print(eval_record[-1])

        return train_record, eval_record

    def _eval_agent(self, i):
        if i > self.config.agent_learn_start:
            return (lambda sm, sp: self.network.act_egreedy(
                self.ms_scaler.transform(sm.reshape(1, -1)).flatten(), sp, e=0.0, device=self.device))
        else:
            return (lambda sm, sp: self.network.act_egreedy(sm, sp, e=0.0, device=self.device))


class Evaluation(object):
    def __init__(self, config):
        super(Evaluation, self).__init__()
        self.config = config
        self.env = make_env(config)

    def evaluate(self, agent):
        bp_list = []
        rew_list = []
        for code in self.config.code_list_validation:
            for date in self.config.date_list_validation:
                record = self.evaluate_single(agent, code=code, date=date)
                bp_list.append(record['BP'].values[-1])
                rew_list.append(record['reward'].sum())

        return dict(
            BP=np.mean(bp_list),
            reward=np.mean(rew_list)
        )

    def evaluate_detail_batch(self, agent, iteration=1,
        code_list=['000504.XSHE', '300750.XSHE', '002466.XSHE', '601899.XSHG'], 
        date_list=['2021-07-21', '2021-07-22', '2021-07-23', '2021-07-26', '2021-07-27', '2021-07-28']):

        path = os.path.join(self.config.result_path, 'evaluation', 'it{:08d}'.format(iteration))
        os.makedirs(path, exist_ok=True)

        record = []
        for date in date_list:
            for code in code_list:
                for i in range(5):
                    res = self.evaluate_single(agent, code=code, date=date)
                    record.append(res)
                    Figure().plot_policy_detail(res,
                        filename=os.path.join(path, 'fig_{}_{}_{}.png'.format(code, date, i)))

        record = pd.concat(record)
        record.to_csv(os.path.join(path, 'detail_{}.csv'.format(code)))

    def evaluate_single(self, agent, code='600519.XSHG', date='2021-06-01'):
        record = []
        sm, sp = self.env.reset(code, date)
        done = False 
        step = 0
        action = None 
        info = dict(status=None)
        pre_quantity = self.env.quantity

        while not done:
            action = agent(sm, sp)
            nsm, nsp, reward, done, info = self.env.step(action)

            if self.config.simulation_action_type == 'discrete_pq':
                order_price = self.config.simulation_discrete_actions[action][0]
                order_price = np.round((1 + order_price / 10000) \
                    * self.env.data.obtain_level('askPrice', 1) * 100) / 100
                order_volume = self.config.simulation_discrete_actions[action][1]
            elif self.config.simulation_action_type == 'discrete_p':
                order_price = self.config.simulation_discrete_actions[action]
                order_price = np.round((1 + order_price / 10000) \
                    * self.env.data.obtain_level('askPrice', 1) * 100) / 100
                order_volume = 0
            elif self.config.simulation_action_type == 'discrete_q':
                order_price = self.env.data.obtain_level('bidPrice', 1)
                order_volume = self.config.simulation_discrete_actions[action]

            record.append(dict(
                code=code,
                date=date,
                datetime=self.env.data.obtain_feature('time'),
                step=step,
                pre_quantity=pre_quantity,
                quantity=self.env.quantity,
                total_quantity=self.env.total_quantity,
                action=action,
                ask_price=self.env.data.obtain_level('askPrice', 1),
                bid_price=self.env.data.obtain_level('bidPrice', 1),
                ask_price_1=self.env.data.obtain_level('askPrice', 1),
                bid_price_1=self.env.data.obtain_level('bidPrice', 1),
                ask_price_2=self.env.data.obtain_level('askPrice', 2),
                bid_price_2=self.env.data.obtain_level('bidPrice', 2),
                ask_price_3=self.env.data.obtain_level('askPrice', 3),
                bid_price_3=self.env.data.obtain_level('bidPrice', 3),
                ask_price_4=self.env.data.obtain_level('askPrice', 4),
                bid_price_4=self.env.data.obtain_level('bidPrice', 4),
                ask_price_5=self.env.data.obtain_level('askPrice', 5),
                bid_price_5=self.env.data.obtain_level('bidPrice', 5),
                order_price=order_price,
                order_volume=order_volume,
                reward=reward,
                cash=self.env.cash,
                BP=self.env.get_metric('BP'),
                IS=self.env.get_metric('IS'),
                status=info['status'],
                index=self.env.data.current_index
            ))
            step += 1
            sm, sp = nsm, nsp
            pre_quantity = self.env.quantity

        return pd.DataFrame(record)


class Figure(object):
    def __init__(self):
        pass

    @staticmethod
    def plot_policy_detail(data, filename):

        cost = data['BP'].values[-1]
        dsize = data.shape[0]
        min_price = data['bid_price_5'].min() - 0.02
        max_price = data['ask_price_5'].max() + 0.02

        start_time = data['datetime'].iloc[0].strftime('%H:%M')
        end_time = data['datetime'].iloc[-1].strftime('%H:%M')
        date = data['datetime'].iloc[0].strftime('%Y-%m-%d')

        total_quantity = data['pre_quantity'].max()
        curr_quantity = total_quantity
        transactions = []
        for i in range(dsize):
            transactions.append(curr_quantity - data.loc[i, 'quantity'])
            curr_quantity = data.loc[i, 'quantity']

        plt.figure(figsize=(15, 6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        for i in range(1, 6):
            ax1.fill_between(np.arange(30) + 1, [min_price]*30, data['bid_price_{}'.format(i)], 
                color='C0', alpha=0.2)
            ax1.fill_between(np.arange(30) + 1, [max_price]*30, data['ask_price_{}'.format(i)], 
                color='C0', alpha=0.2)

        index = np.arange(30) + 1
        order_price = data['order_price']
        select = (data['status'] == 'NOT_FILLED')
        ax1.plot(index[select], order_price[select], 'C3*', ms=10, label='Not Filled Order')
        select = (data['status'] != 'NOT_FILLED')
        ax1.plot(index[select], order_price[select], 'C2*', ms=10, label='Filled/Partially Filled Order')
        ax1.plot([0, 0], [0, 0], 'C1', lw=3, label='Remaining Inventory')
        ax1.set_ylim([min_price, max_price])
        ax1.set_xlim([0.9, 30.1])
        ax1.set_ylabel('Price', fontsize=15)
        ax1.set_xticks([])
        ax1.legend(fontsize=15)
        ax2.plot(list(range(1, 31)) + [30], [total_quantity] + data['quantity'].tolist(), 'C1', lw=3)
        ax2.axis('off')
        ax2.set_ylim([-0.01 * total_quantity, total_quantity * 1.01])
        ax1.set_title('000504.XSHE on {} {} - {} cost={:.4f}'.format(date, start_time, end_time, cost), fontsize=15)
        ax1.set(frame_on=False)

        plt.savefig(filename, bbox_inches='tight')
        plt.close('all')

    @staticmethod
    def plot_policy(df, filename):
        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax2 = ax1.twinx()
        ax1.plot(df['index'], df['ask_price'], label='ask_price')
        ax1.plot(df['index'], df['bid_price'], label='bid_price')
        ax1.plot(df['index'], df['order_price'], label='order_price')
        ax1.legend(loc='lower left')
        ax2.plot(df['index'], df['quantity'], 'k*', label='inventory')
        ax1.set_title('{} {} BP={:.4f}'.format(df['code'].values[-1], df['date'].values[-1], df['BP'].values[-1]))
        ax2.legend(loc='upper right')
        plt.savefig(filename, bbox_inches='tight')
        plt.close('all')

    @staticmethod
    def plot_training_process_basic(df, filename):
        while df.shape[0] > 1500:
            df = df[::2]
        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax2 = ax1.twinx()
        ax1.plot(df.index.values, df['reward'], 'C0', label='reward')
        ax1.legend(loc='lower left')
        ax2.plot(df.index.values, df['BP'], 'C1', label='BP')
        ax2.legend(loc='upper right')
        top_size = df.shape[0] // 10
        mean_bp_first = np.mean(df['BP'].values[:top_size])
        mean_bp_last = np.mean(df['BP'].values[-top_size:])
        mean_rew_first = np.mean(df['reward'].values[:top_size])
        mean_rew_last = np.mean(df['reward'].values[-top_size:])
        ax2.set_title('BP {:.4f}->{:.4f} reward {:.4f}->{:.4f}'.format(mean_bp_first, mean_bp_last, mean_rew_first, mean_rew_last))

        if 'loss' in df.columns:
            ax3 = ax1.twinx()
            p3, = ax3.plot(df.index.values, df['loss'], 'C2')
            ax3.yaxis.label.set_color('C2')

        if 'loss_policy' in df.columns:
            ax3 = ax1.twinx()
            p3, = ax3.plot(df.index.values, df['loss_policy'], 'C3')
            ax3.yaxis.label.set_color('C3')

        if 'loss_encoder' in df.columns:
            ax3 = ax1.twinx()
            p3, = ax3.plot(df.index.values, df['loss_encoder'], 'C4')
            ax3.yaxis.label.set_color('C4')

        plt.savefig(filename, bbox_inches='tight')
        plt.close('all')
        return dict(mean_bp_first=mean_bp_first, mean_bp_last=mean_bp_last, mean_rew_first=mean_rew_first, mean_rew_last=mean_rew_last)

def run(argus):

    mode, feature_set, model, lr, lin_reg, parallel_id = argus

    config = DefaultConfig()
    config.simulation_features = feature_set
    config.agent_learning_rate = lr
    config.simulation_linear_reg_coeff = lin_reg
    config.agent_network_structrue = model
    feature_set_name = 'LOB' if len(feature_set) <= 27 else 'FULL'

    info = dict(mode=mode, feature_set=feature_set_name, learning_rate=lr, 
        linear_reg=lin_reg, architecture=model.__name__, parallel_id=parallel_id)

    id_str = '{}_{}_{}_lr{:.1E}_linreg_{:.1E}_{}'.format(mode, feature_set_name, model.__name__, lr, lin_reg, parallel_id)
    config.result_path = os.path.join(config.result_path, id_str)
    os.makedirs(config.result_path, exist_ok=True)
    extend_path = lambda x: os.path.join(config.result_path, x)

    agent = Agent(config, pid=os.getpid())
    try:

        if mode == 'separate':
            encoder_record, train_record, eval_record = agent.learn_separate()
            encoder_record, train_record, eval_record = pd.DataFrame(encoder_record), pd.DataFrame(train_record), pd.DataFrame(eval_record)
            encoder_record.to_csv(extend_path('encoder_record.csv'))
            train_record.to_csv(extend_path('dqn_train_record.csv'))
            eval_record.to_csv(extend_path('dqn_eval_record.csv'))
        elif mode == 'simutaneous':
            train_record, eval_record = agent.learn_simutaneous()
            train_record, eval_record = pd.DataFrame(train_record), pd.DataFrame(eval_record)
            train_record.to_csv(extend_path('dqn_train_record.csv'))
            eval_record.to_csv(extend_path('dqn_eval_record.csv'))
    
    except Exception as e:
        print(id_str)
        print(os.getpid())
        print(agent.device)
        print(e)

    train_info = Figure().plot_training_process_basic(train_record, extend_path('dqn_train_record.png'))
    eval_info = Figure().plot_training_process_basic(eval_record, extend_path('dqn_eval_record.png'))
    info.update({('trn_' + k): v for k, v in train_info.items()})
    info.update({('val_' + k): v for k, v in eval_info.items()})

    return info


def run_best():

    mode = 'separate'
    feature_set = FEATURE_SET_LOB
    model = MLPNetworkLarge
    lr = 1e-4
    lin_reg = 0.1

    config = DefaultConfig()
    config.simulation_features = feature_set
    config.agent_learning_rate = lr
    config.simulation_linear_reg_coeff = lin_reg
    config.agent_network_structrue = model
    feature_set_name = 'LOB' if len(feature_set) <= 27 else 'FULL'

    def _run_single(parallel_id):
        id_str = str(parallel_id)
        config.result_path = os.path.join(config.result_path, id_str)
        os.makedirs(config.result_path, exist_ok=True)
        extend_path = lambda x: os.path.join(config.result_path, x)
        agent = Agent(config, pid=os.getpid())
        encoder_record, train_record, eval_record = agent.learn_separate()
        encoder_record, train_record, eval_record = \
            pd.DataFrame(encoder_record), pd.DataFrame(train_record), pd.DataFrame(eval_record)
        encoder_record.to_csv(extend_path('encoder_record.csv'))
        train_record.to_csv(extend_path('dqn_train_record.csv'))
        eval_record.to_csv(extend_path('dqn_eval_record.csv'))

    pool = Pool(4)
    pool.map(_run_single, np.arange(4))

def run_sweep():

    record = []
    test_list = list(itertools.product(
        ['simutaneous', 'separate'], 
        [FEATURE_SET_FULL, FEATURE_SET_LOB], [MLPNetworkLarge, MLPNetwork], 
        [1e-4, 1e-5], [1.0, 0.1, 0.01], 
        np.arange(5)
    ))
    pool = Pool(9)
    record = pool.map(run, test_list)
    record = pd.DataFrame(record)
    record.to_csv(os.path.join(DefaultConfig().result_path, 'result_original.csv'))
    stats = record.groupby(['mode', 'feature_set', 'learning_rate', 'linear_reg', 'architecture']).agg([np.mean, np.std])
    stats.to_csv(os.path.join(DefaultConfig().result_path, 'result_stats.csv'))

if __name__ == '__main__':
    run_best()
