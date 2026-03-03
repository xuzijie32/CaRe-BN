import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import copy
import torch.nn as nn
import torch.nn.functional as F
NEURON_VTH = 0.5

def batch_norm_update(X, moving_mean, moving_var, num1, num2):
    assert len(X.shape) in (2, 4)
    if len(X.shape) == 2:

        mean = X.mean(dim=0)
        var = ((X - mean) ** 2).mean(dim=0)
    else:
        mean = X.mean(dim=(0, 2, 3), keepdim=True)
        var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    moving_var = num1 / (num1 + num2) * moving_var + num2 / (num1 + num2) * var + num1*num2/(num1 + num2)/(num1+num2)*torch.square(moving_mean-mean)
    moving_mean = num1/(num1 + num2 ) * moving_mean + num2/(num1 + num2)*mean
    return moving_mean, moving_var

class BatchNorm(nn.Module):

    def __init__(self, num_features,spike_ts, momentum=0.8, num_dims=2):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape)*NEURON_VTH/2.0)
        self.beta = nn.Parameter(torch.zeros(shape)+NEURON_VTH/2.0)

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
        self.momentum = momentum

    def forward(self, X, update, re_calibration):
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

        X_hat_list = []
        Y_list = []

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
                self.p_mean = self.p_mean * self.momentum + (1-self.momentum) * torch.square(delta_mean)
                self.p_var = self.p_var * self.momentum + (1-self.momentum) * torch.square(delta_var)
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

                X_hat_list.append(X_hat_step)
                Y_list.append(Y_step)
        Y = torch.stack(Y_list, dim=2)
        return Y

