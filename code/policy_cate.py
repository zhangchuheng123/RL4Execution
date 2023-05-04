"""
Algo2: End-to-end learn a compact context representation
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
    result_path = 'results/exp39'

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
    # Stack the features of the future x bars
    simulation_lookforward_horizon = 30 + 1
    # Whether return flattened or stacked features of the past x bars
    simulation_do_feature_flatten = True
    simulation_direction = 'sell'
    # If the quantity is not fully filled at the last time step, 
    #   we place an MO to liquidate and further plus a penalty
    simulation_not_filled_penalty_bp = 2.0
    # Scale the price delta if we use continuous actions (deprecated)
    simulation_continuous_action_scale = 10
    # ############################### END ###############################

    # ############################### Sweep Parameters ###############################
    simulation_linear_reg_coeff = 0.1
    agent_learning_rate = 1e-5
    simulation_features = FEATURE_SET_FULL
    agent_network_structrue = 'MLPNetworkLarge'
    agent_dim_emb = 8
    agent_network_scheme = 'scheme1'
    agent_loss1 = '1,2'
    agent_loss2 = '1,2'
    agent_loss3 = '1,2'
    # ############################### END ###############################


class MLPNetwork(nn.Module):
    def __init__(self, dim_input1, dim_input2, dim_embedding, dim_output, market_state3_indices, hidden=128):
        super(MLPNetwork, self).__init__()
        
        self.dim_input1 = dim_input1
        self.dim_input2 = dim_input2
        self.dim_input3 = len(market_state3_indices)
        self.dim_embedding = dim_embedding
        self.dim_output = dim_output
        self.market_state3_indices = market_state3_indices
        
        self.fc_enc_1 = nn.Linear(dim_input1, 2 * hidden)
        self.fc_enc_2 = nn.Linear(2 * hidden, hidden)
        self.fc_enc_3 = nn.Linear(dim_input2, hidden)
        self.fc_enc_4 = nn.Linear(2 * hidden, dim_embedding)

        self.fc_pol_51 = nn.Linear(dim_embedding, hidden)
        self.fc_pol_52 = nn.Linear(dim_input2, hidden)
        self.fc_pol_53 = nn.Linear(self.dim_input3, hidden)
        self.fc_pol_6 = nn.Linear(3 * hidden, hidden)
        self.fc_pol_7 = nn.Linear(hidden, dim_output)

    def encoding(self, market_states, private_states):
        x = F.relu(self.fc_enc_1(market_states))
        x = F.relu(self.fc_enc_2(x))
        y = F.relu(self.fc_enc_3(private_states))
        z = torch.cat((x, y), 1)
        z = self.fc_enc_4(z)
        return z
        
    def forward(self, market_states, private_states, detach=True):
        emb = self.encoding(market_states, private_states)
        if detach:
            emb = emb.detach()
        return self._forward_emb(emb, market_states, private_states)

    def _forward_emb(self, emb, market_states, private_states):
        x = F.relu(self.fc_pol_51(emb))
        y = F.relu(self.fc_pol_52(private_states))
        market_states3 = market_states[:, self.market_state3_indices]
        z = F.relu(self.fc_pol_53(market_states3))
        out = torch.cat((x, y, z), 1)
        out = F.relu(self.fc_pol_6(out))
        out = self.fc_pol_7(out)
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
        self.fc_enc_m1 = nn.Linear(dim_input1, hidden)
        self.fc_enc_m2 = nn.Linear(hidden, hidden)
        self.fc_enc_p1 = nn.Linear(dim_input2, hidden)
        self.fc_enc_3 = nn.Linear(2 * hidden, hidden)
        self.fc_enc_4 = nn.Linear(hidden, dim_embedding)

        self.fc_pol_e1 = nn.Linear(dim_embedding, hidden)
        self.fc_pol_p1 = nn.Linear(dim_input2, hidden)
        self.fc_pol_m1 = nn.Linear(self.dim_input3, hidden)
        self.fc_pol_2 = nn.Linear(3 * hidden, hidden)
        self.fc_pol_3 = nn.Linear(hidden, hidden)
        self.fc_pol_4 = nn.Linear(hidden, dim_output)

    def encoding(self, market_states, private_states):
        x = F.relu(self.fc_enc_m1(market_states))
        x = F.relu(self.fc_enc_m2(x))
        y = F.relu(self.fc_enc_p1(private_states))
        z = torch.cat((x, y), 1)
        z = F.relu(self.fc_enc_3(z))
        z = self.fc_enc_4(z)
        return z
        
    def forward(self, market_states, private_states, detach=True):
        emb = self.encoding(market_states, private_states)
        if detach: 
            emb = emb.detach()
        return self._forward_emb(emb, market_states, private_states)

    def _forward_emb(self, emb, market_states, private_states):
        x = F.relu(self.fc_pol_e1(emb))
        y = F.relu(self.fc_pol_p1(private_states))
        market_states3 = market_states[:, self.market_state3_indices]
        z = F.relu(self.fc_pol_m1(market_states3))
        out = torch.cat((x, y, z), 1)
        out = F.relu(self.fc_pol_2(out))
        out = F.relu(self.fc_pol_3(out))
        out = self.fc_pol_4(out)
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
        self.dim_input_future = len(self.config.simulation_features) * self.config.simulation_lookforward_horizon
        self.dim_emb = self.config.agent_dim_emb
        self.market_state3_indices = \
            [config.simulation_features.index(item) for item in config.agent_policy_selected_features]

        # a DQN network that output Q value based on past observations
        network = config.agent_network_structrue
        self.network = network(self.dim_input1, self.dim_input2, self.dim_emb, self.dim_output, self.market_state3_indices)\
            .to(device=self.device)
        self.network_target = network(self.dim_input1, self.dim_input2, self.dim_emb, self.dim_output, self.market_state3_indices)\
            .to(device=self.device)
        self.network_target.load_state_dict(self.network.state_dict())

        # a DQN network that output Q value based on future observations
        self.network_fut = network(self.dim_input_future, self.dim_input2, self.dim_emb, self.dim_output, self.market_state3_indices)\
            .to(device=self.device)
        if self.config.agent_network_scheme == 'scheme2':
            self._share_policy()
        self.network_fut_target = network(self.dim_input_future, self.dim_input2, self.dim_emb, self.dim_output, self.market_state3_indices)\
            .to(device=self.device)
        self.network_fut_target.load_state_dict(self.network_fut.state_dict())

        # loss 3: regularize prediction and label
        self.total_parameters = list(self.network.parameters()) + list(self.network_fut.parameters())
        self.optimizer = opt.Adam(self.total_parameters, lr=config.agent_learning_rate)
        self.scheduler = opt.lr_scheduler.StepLR(self.optimizer, step_size=config.agent_lr_decay_freq, gamma=0.998)

        self.buffer = ReplayBuffer(self.config.agent_buffer_size)
        self.evaluation = Evaluation(self.config)
        if config.agent_loss_type == 'MSE':
            self.loss_func = nn.MSELoss()
        elif config.agent_loss_type == 'SL1':
            self.loss_func = F.smooth_l1_loss
        self.ms_scaler = StandardScaler()

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

    def _share_policy(self):
        for param1, param2 in zip(list(self.network.named_parameters()), list(self.network_fut.named_parameters())):
            if 'pol' in param1[0]:
                param2[1].data = param1[1].data

    def _cross_policy(self):
        for param1, param2 in zip(list(self.network.named_parameters()), list(self.network_fut.named_parameters())):
            if 'pol' in param1[0]:
                target = (param1[1].data + param2[1].data) / 2
                param1[1].data = target
                param2[1].data = target

    @staticmethod
    def _filter(state):
        return np.clip(state, -3, 3)

    def _to_tensor(self, tensor, dtype=torch.float):
        return torch.tensor(tensor, dtype=dtype, device=self.device)

    def _clip_grad(self, val=1):
        for param in self.total_parameters:
            if param.grad is not None:
                param.grad.data.clamp_(-val, val)

    def learn(self):

        train_record = []
        eval_record = []

        loss = loss1 = loss2 = loss3 = 0
        monitor = dict(
            reward=0,
            eplen=0,
            avg_Q1=0,
            avg_Q2=0,
            epsilon=self.config.agent_epsilon
        )
        get_future_func = \
            lambda: self.env.get_future(self.config.simulation_features, \
                padding=self.config.simulation_lookforward_horizon).flatten()

        sm, sp = self.env.reset()
        future = get_future_func()

        for i in trange(self.config.agent_total_steps):

            # Step 1: Execute one step and store it to the replay buffer
            if i <= self.config.agent_learn_start:
                a = self.env.action_sample_func()
            else:
                tsm = self.ms_scaler.transform(sm.reshape(1, -1)).flatten()
                a = self.network.act_egreedy(tsm, sp, e=monitor['epsilon'], device=self.device)

            nsm, nsp, r, done, info = self.env.step(a)
            self.buffer.push(sm, sp, a, r, nsm, nsp, done, future)
            monitor['reward'] += r
            monitor['eplen'] += 1
            if done:
                train_record.append(dict(
                    i=i, 
                    reward=monitor['reward'],
                    eplen=monitor['eplen'],
                    epsilon=monitor['epsilon'],
                    lr=self.optimizer.param_groups[0]['lr'],
                    loss1=float(loss1),
                    loss2=float(loss2),
                    loss3=float(loss3),
                    loss=float(loss),
                    avg_Q1=monitor['avg_Q1'],
                    avg_Q2=monitor['avg_Q2'],
                    BP=self.env.get_metric('BP'),
                    IS=self.env.get_metric('IS'),
                    code=info['code'],
                    date=info['date'],
                    start_index=info['start_index']
                    ))
                monitor['reward'] = 0
                monitor['eplen'] = 0
                monitor['epsilon'] = max(0.01, monitor['epsilon'] * self.config.agent_epsilon_decay)
                sm, sp = self.env.reset()
            else:
                sm, sp = nsm, nsp
            future = get_future_func()

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
                bsm, bsp, ba, br, bnsm, bnsp, bd, bfuture = self.buffer.sample(self.config.agent_batch_size)

                market_states = self._to_tensor(self._filter(self.ms_scaler.transform(np.array(bsm))))
                private_states = self._to_tensor(np.array(bsp))
                actions = self._to_tensor(np.array(ba), dtype=torch.long)
                rewards = self._to_tensor(np.array(br))
                nmarket_states = self._to_tensor(self._filter(self.ms_scaler.transform(np.array(bnsm))))
                nprivate_states = self._to_tensor(np.array(bnsp))
                masks = self._to_tensor(1 - np.array(bd) * 1)
                nactions = self.network(nmarket_states, nprivate_states).argmax(1)
                fut_market_states = self._to_tensor(np.array(bfuture))

                # Update the encoder
                embedding1 = self.network.encoding(market_states, private_states)
                embedding2 = self.network_fut.encoding(fut_market_states, private_states)
                if '1' in self.config.agent_loss3 and '2' in self.config.agent_loss3:
                    loss3 = self.loss_func(embedding1, embedding2)
                elif '1' not in self.config.agent_loss3:
                    loss3 = self.loss_func(embedding1.detach(), embedding2)
                elif '2' not in self.config.agent_loss3:
                    loss3 = self.loss_func(embedding1, embedding2.detach())

                # Update the policy
                if '1' not in self.config.agent_loss1:
                    embedding1 = embedding1.detach()

                if '1' not in self.config.agent_loss2:
                    embedding2 = embedding2.detach()

                Qtarget = (rewards + masks * self.config.agent_gamma * \
                    self.network_target(nmarket_states, nprivate_states)\
                    [range(self.config.agent_batch_size), nactions]).detach()

                Qvalue1 = self.network._forward_emb(embedding1, market_states, private_states)\
                    [range(self.config.agent_batch_size), actions]
                monitor['avg_Q1'] = Qvalue1.mean().detach()
                Qvalue2 = self.network_fut._forward_emb(embedding2, market_states, private_states)\
                    [range(self.config.agent_batch_size), actions]
                monitor['avg_Q2'] = Qvalue2.mean().detach()

                loss1 = self.loss_func(Qvalue1, Qtarget)
                loss2 = self.loss_func(Qvalue2, Qtarget)

                loss = loss1 + (0.5 * loss2) + loss3

                self.network.zero_grad()
                loss.backward()
                self._clip_grad()
                self.optimizer.step()
                self.scheduler.step()

                if self.config.agent_network_scheme == 'scheme2':
                    self._cross_policy()
                
            # Step 5: Update target network
            if i % self.config.agent_target_update_freq == 0:
                self.network_target.load_state_dict(self.network.state_dict())
                self.network_fut_target.load_state_dict(self.network_fut.state_dict())

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
        code='000504.XSHE', 
        date_list=['2021-06-01', '2021-06-03', '2021-06-04', '2021-07-02', '2021-07-05', '2021-07-06']):

        path = os.path.join(self.config.result_path, 'evaluation', 'it{:08d}'.format(iteration))
        os.makedirs(path, exist_ok=True)

        record = []
        for date in date_list:
            for i in range(5):
                res = self.evaluate_single(agent, code=code, date=date)
                record.append(res)
                Figure().plot_policy(df=res, filename=os.path.join(path, 'fig_{}_{}_{}.png'.format(code, date, i)))

        pd.concat(record).to_csv(os.path.join(path, 'detail_{}.csv'.format(code)))

    def evaluate_single(self, agent, code='600519.XSHG', date='2021-06-01'):
        record = []
        sm, sp = self.env.reset(code, date)
        done = False 
        step = 0
        action = None 
        info = dict(status=None)

        while not done:
            action = agent(sm, sp)
            nsm, nsp, reward, done, info = self.env.step(action)

            if self.config.simulation_action_type == 'discrete_pq':
                order_price = self.config.simulation_discrete_actions[action][0]
                order_price = np.round((1 + order_price / 10000) \
                    * self.env.data.obtain_level('askPrice', 1) * 100) / 100
            elif self.config.simulation_action_type == 'discrete_p':
                order_price = self.config.simulation_discrete_actions[action]
                order_price = np.round((1 + order_price / 10000) \
                    * self.env.data.obtain_level('askPrice', 1) * 100) / 100
            elif self.config.simulation_action_type == 'discrete_q':
                order_price = self.env.data.obtain_level('bidPrice', 1)

            record.append(dict(
                code=code,
                date=date,
                step=step,
                quantity=self.env.quantity,
                action=action,
                ask_price=self.env.data.obtain_level('askPrice', 1),
                bid_price=self.env.data.obtain_level('bidPrice', 1),
                order_price=order_price,
                reward=reward,
                cash=self.env.cash,
                BP=self.env.get_metric('BP'),
                IS=self.env.get_metric('IS'),
                status=info['status'],
                index=self.env.data.current_index
            ))
            step += 1
            sm, sp = nsm, nsp

        return pd.DataFrame(record)


class Figure(object):
    def __init__(self):
        pass

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

    scheme, dim_emb, loss1, loss2, loss3, model, parallel_id = argus

    config = DefaultConfig()
    config.agent_network_scheme = scheme
    config.agent_dim_emb = dim_emb
    config.agent_loss1 = loss1
    config.agent_loss2 = loss2
    config.agent_loss3 = loss3 
    config.agent_network_structrue = model

    info = dict(scheme=scheme, loss1=loss1, loss2=loss2, loss3=loss3, dim_emb=dim_emb,
        architecture=model.__name__, parallel_id=parallel_id)

    id_str = '{}_{}_{}_{}_{}_{}_{}'.format(scheme, dim_emb, loss1, loss2, loss3, model.__name__, parallel_id)
    config.result_path = os.path.join(config.result_path, id_str)
    os.makedirs(config.result_path, exist_ok=True)
    extend_path = lambda x: os.path.join(config.result_path, x)

    agent = Agent(config, pid=os.getpid())
    train_record, eval_record = agent.learn()
    train_record, eval_record = pd.DataFrame(train_record), pd.DataFrame(eval_record)
    train_record.to_csv(extend_path('ours_train_record.csv'))
    eval_record.to_csv(extend_path('ours_eval_record.csv'))

    train_info = Figure().plot_training_process_basic(train_record, extend_path('dqn_train_record.png'))
    eval_info = Figure().plot_training_process_basic(eval_record, extend_path('dqn_eval_record.png'))
    info.update({('trn_' + k): v for k, v in train_info.items()})
    info.update({('val_' + k): v for k, v in eval_info.items()})

    return info

def extract_info(path):
    folders = os.listdir(path)
    record = []
    for folder in folders:
        print('processing {}'.format(folder))
        scheme, dim_emb, loss1, loss2, loss3, model, parallel_id = folder.split('_')
        info = dict(scheme=scheme, loss1=loss1, loss2=loss2, loss3=loss3, dim_emb=dim_emb,
            architecture=model, parallel_id=parallel_id)

        if os.path.exists(os.path.join(path, folder, 'ours_train_record.csv')):
            train_record = pd.read_csv(os.path.join(path, folder, 'ours_train_record.csv'), index_col=0)
            eval_record = pd.read_csv(os.path.join(path, folder, 'ours_eval_record.csv'), index_col=0)
            train_info = Figure().plot_training_process_basic(train_record, 'tmp.png')
            eval_info = Figure().plot_training_process_basic(eval_record, 'tmp.png')
            info.update({('trn_' + k): v for k, v in train_info.items()})
            info.update({('val_' + k): v for k, v in eval_info.items()})
            
        record.append(info)

    record = pd.DataFrame(record)
    record.to_csv(os.path.join(DefaultConfig().result_path, 'result_original.csv'))
    stats = record.groupby(['scheme', 'loss1', 'loss2', 'loss3', 'dim_emb', 'architecture']).agg([np.mean, np.std])
    stats.to_csv(os.path.join(DefaultConfig().result_path, 'result_stats.csv'))

if __name__ == '__main__':

    # Test run
    # run(['scheme2', 4, '1,2', '1,2', '1,2', MLPNetworkLarge, 0])
    # run(['scheme2', 4, '1,2', '1,2', '1,2', MLPNetwork, 0])

    # Main run 
    record = []
    test_list = list(itertools.product(
        ['scheme1', 'scheme2'],
        [4, 8, 16],
        ['1,2', '2'],
        ['1,2'],
        ['1,2', '1'],
        [MLPNetworkLarge, MLPNetwork], 
        np.arange(3)
    ))
    pool = Pool(16)
    record = pool.map(run, test_list)
    record = pd.DataFrame(record)
    record.to_csv(os.path.join(DefaultConfig().result_path, 'result_original.csv'))
    stats = record.groupby(['scheme', 'loss1', 'loss2', 'loss3', 'dim_emb', 'architecture']).agg([np.mean, np.std])
    stats.to_csv(os.path.join(DefaultConfig().result_path, 'result_stats.csv'))
