import numpy as np   
import texas_holdem_mod as texas_holdem
import rlcard 
from rlcard.utils.utils import print_card as prnt_cd
import texas_holdem_mod as texas_holdem
import rlcard 
from rlcard.utils.utils import print_card as prnt_cd
from stable_baselines3.common.env_checker import check_env
from evaluation_mod import evaluate_policy
from callbacks_mod import EvalCallback
from callbacks_mod import StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from callbacks_mod import BaseCallback
import os 
import matplotlib.pyplot as plt
from gymnasium import Env
import optuna
import gym
import numpy as np
import torch as th
from torch import nn
from tabulate import tabulate
import pandas as pd
from rlcard.agents.human_agents.nolimit_holdem_human_agent import HumanAgent
from classmaker import graph_metrics
from classmaker import obs_type_envs
from classmaker import metric_dicts
from ppo import PPO
class human_play_callback(BaseCallback):
    def __init__(self, verbose=0):
        super(human_play_callback, self).__init__(verbose)
        self.losses = []
        self.moving_mean_reward = []
        self.rewards = [] 
        self.moving_total = [0]
        self.step_list = []
        
        # self.op_moving_mean_reward= []
        # self.op_rewards = []
        # self.op_moving_total = [0]
        if len( self.moving_mean_reward)>1:
            self.final_mean_reward = self.moving_mean_reward[-1]
        
    def _on_step(self) -> bool:
        if hasattr(self.model, 'loss'):
            if self.model.loss != None:
                loss = self.model.loss.item()
                loss = float(loss)
                self.losses.append(loss)     
        self.step_list.append(self.num_timesteps)
        self.rewards.append(self.model.env.buf_rews[0])
        self.moving_mean_reward.append(np.mean(self.rewards))
        self.moving_total.append(self.moving_total[-1] + self.model.env.buf_rews[0])
        if len( self.moving_mean_reward)>1:
            self.final_mean_reward = self.moving_mean_reward[-1]

class human_play():
    def __init__(self,obs_type, n_eval, root):    
        self.obs_type = obs_type
        self.n_eval_episodes = n_eval
        self.root = root
        self.set_up_env()

    def set_up_env(self):
        # root = '/Users/rhyscooper/Desktop/MSc Project/Pages/models/1002_6.zip'
        Eval_env = texas_holdem.env(self.obs_type, render_mode = "rgb_array")
        Eval_env.AGENT.policy = 'PPO'
        Eval_env.AGENT.model = PPO('MultiInputPolicy', Eval_env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [256], 'vf': [256]},learning_rate= 0.005778633008004902, n_steps = 3072,  batch_size = 32, n_epochs= 70, ent_coef=  0.0025, vf_coef=  0.25, clip_range=0.1, max_grad_norm=0.6, gae_lambda = 0.85, normalize_advantage=False)
        Eval_env.AGENT.model.set_parameters(load_path_or_dict= self.root)
        Eval_env.OPPONENT.policy = 'human'
        
        self.Eval_env = Monitor(Eval_env)
        self.env = Eval_env
        
    def play(self):
        mean_reward,episode_rewards = evaluate_policy(self.env.AGENT.model, self.Eval_env, n_eval_episodes = self.n_eval_episodes, verbose = True, return_episode_rewards= False)
        self.mean_reward = mean_reward  

        

        
