import os
from typing import Dict, List, Tuple
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
import math
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import cv2
import ale_py



env_name = "Pong-v5"
seed = 0

lr=0.0001
start_training = 10000
num_frames = 2000000 + start_training
eval_frequency = 10000
memory_size = 100000
batch_size = 128
target_update = 2000
epsilon_decay = 1 / 200000
SG = "rect"

re_calibration_time=100



ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
NEURON_CDECAY = 0
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5

def batch_norm_update(X, moving_mean, moving_var, num1, num2):
    assert len(X.shape) in (2, 4)
    if len(X.shape) == 2:
        # 使用全连接层的情况，计算特征维上的均值和方差
        mean = X.mean(dim=0)
        var = ((X - mean) ** 2).mean(dim=0)
    else:
        # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。
        # 这里我们需要保持X的形状以便后面可以做广播运算
        mean = X.mean(dim=(0, 2, 3), keepdim=True)
        var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    moving_var = num1 / (num1 + num2) * moving_var + num2 / (num1 + num2) * var + num1*num2/(num1 + num2)/(num1+num2)*torch.square(moving_mean-mean)
    moving_mean = num1/(num1 + num2 ) * moving_mean + num2/(num1 + num2)*mean
    return moving_mean, moving_var

class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features,spike_ts, momentum=0.9, num_dims=2):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = nn.Parameter(torch.ones(shape)*NEURON_VTH/2.0)
        self.beta = nn.Parameter(torch.zeros(shape)+NEURON_VTH/2.0)
        # 非模型参数的变量初始化为0和1
        self.moving_mean = nn.Parameter(torch.zeros(shape),requires_grad=False)
        self.moving_var = nn.Parameter(torch.ones(shape),requires_grad=False)
        self.temp_mean = torch.zeros(shape)
        self.temp_var = torch.ones(shape)
        self.p_mean = torch.zeros(shape)
        self.p_var = torch.zeros(shape)
        self.K_mean = torch.zeros(shape)
        self.K_var = torch.zeros(shape)
        self.spike_ts=spike_ts
        self.eps=1e-5
        self.bs=batch_size

    def forward(self, X, update, re_calibration):
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        if self.temp_mean.device != X.device:
            self.temp_mean = self.temp_mean.to(X.device)
            self.temp_var = self.temp_var.to(X.device)
        if self.p_mean.device != X.device:
            self.p_mean = self.p_mean.to(X.device)
            self.p_var = self.p_var.to(X.device)
        if self.K_mean.device != X.device:
            self.K_mean = self.K_mean.to(X.device)
            self.K_var = self.K_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        X_hat_list = []
        Y_list = []

        # aaa=X.shape[0]
        # breakpoint()
        if re_calibration:
            with torch.no_grad():
                if re_calibration[0]==0:
                    self.temp_mean = torch.zeros_like(self.temp_mean)
                    self.temp_var = torch.zeros_like(self.temp_var)
                for step in range(self.spike_ts):
                    self.temp_mean, self.temp_var = batch_norm_update(X[:, :, step], self.temp_mean, self.temp_var, step + re_calibration[0] * self.spike_ts, 1)

                if re_calibration[1]==re_calibration[0]+1:
                    self.moving_mean = nn.Parameter(self.temp_mean,requires_grad=False)
                    self.moving_var = nn.Parameter(self.temp_var,requires_grad=False)


                for step in range(self.spike_ts):
                    X_hat_step = (X[:, :, step] - self.temp_mean) / torch.sqrt(self.temp_var + self.eps)
                    Y_step = self.gamma * X_hat_step + self.beta

                    # 将结果添加到列表
                    X_hat_list.append(X_hat_step)
                    Y_list.append(Y_step)

        elif update:
            self.temp_mean=torch.zeros_like(self.temp_mean)
            self.temp_var=torch.zeros_like(self.temp_var)
            for step in range(self.spike_ts):
                self.temp_mean,self.temp_var=batch_norm_update(X[:,:,step], self.temp_mean, self.temp_var, step, 1)
            
            with torch.no_grad():
                delta_mean = self.temp_mean - self.moving_mean
                delta_var = self.temp_var - self.moving_var
                mean_var = self.temp_var / 255
                var_var = 2 * torch.square(self.temp_var) / 255
                self.p_mean = self.p_mean * 0.8 + 0.2 * torch.square(delta_mean)
                self.p_var = self.p_var * 0.8 + 0.2 * torch.square(delta_var)
                self.K_mean = self.p_mean / (self.p_mean + mean_var)
                self.K_var = self.p_var / (self.p_var + var_var)
                self.moving_mean = nn.Parameter(self.moving_mean + self.K_mean * delta_mean, requires_grad=False)
                self.moving_var = nn.Parameter(self.moving_var + self.K_var * delta_var, requires_grad=False)
            
            for step in range(self.spike_ts):
                X_hat_step = (X[:, :, step] - self.temp_mean) / torch.sqrt(self.temp_var + self.eps)
                Y_step = self.gamma * X_hat_step + self.beta

                # 将结果添加到列表
                X_hat_list.append(X_hat_step)
                Y_list.append(Y_step)
        else:
            for step in range(self.spike_ts):
                X_hat_step = (X[:, :, step] - self.moving_mean) / torch.sqrt(self.moving_var + self.eps)
                Y_step = self.gamma * X_hat_step + self.beta

                # 将结果添加到列表
                X_hat_list.append(X_hat_step)
                Y_list.append(Y_step)
        Y = torch.stack(Y_list, dim=2)
        return Y


class PopSpikeEncoderRegularSpike(nn.Module):
    """ Learnable Population Coding Spike Encoder with Regular Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        self.device = device

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # Generate Regular Spike Trains
        for step in range(self.spike_ts):
            pop_spikes[:, :, step] = obs
        return pop_spikes


class PseudoSpikeRect(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        if SG == 'rect':
            spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        elif SG == 'sigmoid':
            y = torch.sigmoid(4 * (input - NEURON_VTH))
            spike_pseudo_grad = 4 * y * (1-y)
        elif SG == 'tanh':
            y = torch.tanh(2 * (input - NEURON_VTH))
            spike_pseudo_grad = 1 - y * y
        elif SG == 'triangle':
            aaa = abs(input - NEURON_VTH)
            bbb = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
            spike_pseudo_grad = (SPIKE_PSEUDO_GRAD_WINDOW - aaa) * bbb.float() / SPIKE_PSEUDO_GRAD_WINDOW / SPIKE_PSEUDO_GRAD_WINDOW
        else:
            raise ValueError('SG must be either "rect", "sigmoid", "tanh" or "triangle"')
        return grad_input * spike_pseudo_grad.float()

class SpikeMLP(nn.Module):
    """ Spike MLP with Input and Output population neurons """
    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device):
        """
        :param in_pop_dim: input population dimension
        :param out_pop_dim: output population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoSpikeRect.apply
        # Define Layers (Hidden Layers + Output Population)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0])])
        self.hidden_norms = nn.ModuleList([BatchNorm(hidden_sizes[0],spike_ts)])
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.extend([nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer])])
                self.hidden_norms.extend([BatchNorm(hidden_sizes[layer],spike_ts)])
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim)
        self.out_pop_norm = BatchNorm(out_pop_dim,spike_ts)

    def neuron_model(self, pre_layer_output, current, volt, spike):
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        current = current * NEURON_CDECAY + pre_layer_output
        volt = volt * (1 - spike) * NEURON_VDECAY + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, in_pop_spikes, batch_size, norm_update, re_calibration):
        """
        :param in_pop_spikes: input population spikes
        :param batch_size: batch size
        :return: out_pop_act
        """
        # Define LIF Neuron states: Current, Voltage, and Spike
        hidden_states = []
        out_spikes = []
        X=[]
        X_=[]
        for layer in range(self.hidden_num):
            hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                  for _ in range(3)])
            out_spikes.append(torch.zeros(batch_size, self.hidden_sizes[layer], self.spike_ts, device=self.device))
            X.append(torch.zeros(batch_size, self.hidden_sizes[layer], self.spike_ts, device=self.device))
            X_.append(torch.zeros(batch_size, self.hidden_sizes[layer], self.spike_ts, device=self.device))
        out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device)
                          for _ in range(3)]
        X.append(torch.zeros(batch_size, self.out_pop_dim, self.spike_ts, device=self.device))
        X_.append(torch.zeros(batch_size, self.out_pop_dim, self.spike_ts, device=self.device))
        out_pop_act = torch.zeros(batch_size, self.hidden_sizes[-1], device=self.device)
        # Start Spike Timestep Iteration
        for step in range(self.spike_ts):
            in_pop_spike_t = in_pop_spikes[:, :, step]
            X[0][:,:,step]=self.hidden_layers[0](in_pop_spike_t)
        X_[0]=self.hidden_norms[0](X[0],update=norm_update, re_calibration=re_calibration)
        for step in range(self.spike_ts):
            hidden_states[0][0], hidden_states[0][1], hidden_states[0][2] = self.neuron_model(X_[0][:,:,step],
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2])
            out_spikes[0][:,:,step]=hidden_states[0][2]
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                for step in range(self.spike_ts):
                    in_pop_spike_t = out_spikes[layer-1][:,:,step]
                    X[layer][:, :, step] = self.hidden_layers[layer](in_pop_spike_t)
                X_[layer]=self.hidden_norms[layer](X[layer],update=norm_update, re_calibration=re_calibration)
                for step in range(self.spike_ts):
                    hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2] = self.neuron_model(X_[layer][:,:,step],
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2])
                    out_spikes[layer][:,:,step]=hidden_states[layer][2]
        
                    if layer == self.hidden_num - 1:
                        out_pop_act += hidden_states[self.hidden_num-1][2]

        Q = self.out_pop_layer(out_pop_act)
        return Q
        

class Network(nn.Module):
    """ Population Coding Spike Actor with Fix Encoder """
    def __init__(self, obs_dim, act_dim, en_pop_dim=1, de_pop_dim=1, hidden_sizes=[128,128],
                 mean_range=(-1,1), std=math.sqrt(0.05), spike_ts=5, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param en_pop_dim: encoder population dimension
        :param de_pop_dim: decoder population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param spike_ts: spike timesteps
        :param act_limit: action limit
        :param device: device
        :param use_poisson: if true use Poisson spikes for encoder
        """
        super().__init__()
        self.encoder = PopSpikeEncoderRegularSpike(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        self.snn = SpikeMLP(obs_dim*en_pop_dim, act_dim*de_pop_dim, hidden_sizes, spike_ts, device)
        self.obs_dim = obs_dim
        self.device = device


    def forward(self, obs, update = False, re_calibration=False):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: action scale with action limit
        """
        aaa = obs.view(-1)
        bbb = aaa.view(-1,self.obs_dim)
        state = bbb
        # state = torch.FloatTensor(np.array(obs).reshape(1, -1)).to(self.device)
        batch_size=state.size()[0]
        in_pop_spikes = self.encoder(state, batch_size)
        Q = self.snn(in_pop_spikes, batch_size, update, re_calibration)
        return Q

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size



class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
            self,
            env: gym.Env,
            eval_env: gym.Env,
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float,
            seed: int,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.eval_env = eval_env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        self.start_training = start_training

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(),lr=lr)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # epsilon greedy policy
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        eval_scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # if episode ends
            if done:
                state, _ = self.env.reset(seed=self.seed)
                scores.append(score)
                score = 0

            # if training is ready
            if frame_idx > self.start_training:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # linearly decrease epsilon
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self.re_calibration()
                eval_score = self.evaluate()
                print("Episode:\t", frame_idx, "\tEvaluation:\t",eval_score)
                eval_scores.append(eval_score)
                np.save(env_name + '-' + str(seed), eval_scores)
        self.env.close()

    def evaluate(self,trials=10):
        score=0
        for trial in range(trials):
            state, _ = self.eval_env.reset(seed=self.seed+100)
            done = False
            while not done:
                action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
                action = action.detach().cpu().numpy()
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated

                state = next_state
                score += reward
        score /= trials
        return score



    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True

        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        # reset
        self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state, update = True).gather(1, action)
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        # loss = F.smooth_l1_loss(curr_q_value, target)
        loss = F.mse_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def re_calibration(self):
        for _ in range(re_calibration_time):
            samples = self.memory.sample_batch()
            state = torch.FloatTensor(samples["obs"]).to(self.device)
            self.dqn(state,re_calibration=[_,re_calibration_time])


env = gym.make("ALE/"+env_name,obs_type='ram')
eval_env = gym.make("ALE/"+env_name,obs_type='ram')


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
seed_torch(seed)

env.reset(seed=seed)
env.action_space.seed(seed)

eval_env.reset(seed=seed + 100)
eval_env.action_space.seed(seed + 100)

agent = DQNAgent(env, eval_env, memory_size, batch_size, target_update, epsilon_decay, seed)
agent.train(num_frames,eval_frequency)
