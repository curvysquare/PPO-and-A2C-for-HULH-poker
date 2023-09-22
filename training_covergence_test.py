from self_play import self_play
import matplotlib.pyplot as plt
import numpy as np   
import texas_holdem_mod as texas_holdem
from rlcard.utils.utils import print_card as prnt_cd
from rlcard.utils.utils import print_card as prnt_cd
from env_checker_mod import check_env
from evaluation_mod import evaluate_policy
from callbacks_mod import EvalCallback
from callbacks_mod import StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
try:
    from stable_baselines3.common.callbacks_mod import BaseCallback
except ModuleNotFoundError:
    from stable_baselines3.common.callbacks import BaseCallback
import os 
import matplotlib.pyplot as plt
from ppo import PPO
from a2c import A2C
from gymnasium import Env
import optuna
import gym
import numpy as np
import torch as th
from torch import nn
from tabulate import tabulate
import pandas as pd
import random
from rlcard.agents.human_agents.nolimit_holdem_human_agent import HumanAgent
from classmaker import graph_metrics
from classmaker import metric_dicts

from injector import card_injector
from human_input import human_play
from self_play import CustomLoggerCallback
class train_convergence_search():
    """
    Class to perform a search for convergence-related hyperparameters in a PPO training setup.

    This class initializes a search for convergence-related hyperparameters using Optuna, specifically
    for training a PPO agent. It defines a custom Optuna callback
    to stop training based on convergence criteria. It also tracks trial results and provides
    optimization parameters for the Optuna study.

    Args:
    - verbose (bool): Whether to display progress bars during training.
    - obs_type (str): Type of observation space in the environment.

    Attributes:
    - verbose (bool): Whether to display progress bars during training.
    - obs_type (str): Type of observation space in the environment.
    - model (str): Type of RL model being trained (default: 'PPO').
    - trial_results (dict): Dictionary to store trial results.

    Methods:
    - init_trained_op(): Initialize the trained opponent models.
    - callback(trial, study): Custom Optuna callback for tracking trial results.
    - optimize_cb_params(trial): Define optimization parameters for the convergence search.
    - optimize_cb(trial): Perform the convergence search using Optuna.
    """

    def __init__(self):
        self.title = "TCS"
        self.env = texas_holdem.env('72', render_mode = "rgb_array")
        self.env.OPPONENT.policy = 'random'
        self.env.AGENT.policy = 'PPO'
        self.metric_dicts = metric_dicts(1)
        self.metric_dicts.add_keys_to_metrics([0])
        self.env.AGENT.model = PPO('MultiInputPolicy', self.env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [64], 'vf': [64]})  

    
        
            
    def run(self, total_timesteps):
        self.total_timesteps = total_timesteps
        callback_train = CustomLoggerCallback()
        self.env.AGENT.model.learn(total_timesteps, dumb_mode = False, progress_bar=True, callback= None)
        self.env.OPPONENT.model = self.env.AGENT.model
        self.env.OPPONENT.policy = 'PPO'
        self.env.AGENT.model = PPO('MultiInputPolicy', self.env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [64], 'vf': [64]})
        self.env.AGENT.model.learn(total_timesteps, dumb_mode = False, progress_bar=True, callback= callback_train)
        self.metric_dicts.update_train_metrics_from_callback(0, callback_train)

    def print_graphs(self):
            gm = graph_metrics(n_models = 1, storage = self.metric_dicts, storageB= None, figsize= (10, 8), t_steps =self.total_timesteps, overlay= False, e_steps=0, title = self.title, device='pc' )
            gm.print_select_graphs(False, False, True, False, False, False, False,True, False, True)

          