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

class kl_div_test():
    """
    A class for testing the Kullback-Leibler (KL) divergence between policies in a Texas Hold'em environment.

    This class initializes multiple pre-trained PPO models and evaluates their KL divergence when used as
    agents in a Texas Hold'em environment.

    Parameters:
    - obs_type (str): The observation type for the Texas Hold'em environment ('rgb_array' or 'tensor').
    - n_gens (int): The number of generations to evaluate.

    Attributes:
    - obs_type (str): The observation type for the environment.
    - results (dict): A dictionary to store the KL divergence results.
    - n_gens (int): The number of generations to evaluate.
    - metric_dicts: An instance of the metric_dicts class to store metric data.
    - gen_keys (list): A list of generation keys for tracking the evaluations.

    Methods:
    - run(): Runs the evaluation process for KL divergence between policies.
    """
    def __init__(self, obs_type, n_gens):
        self.obs_type = obs_type
        self.results = {}
        self.n_gens = n_gens
        self.metric_dicts = metric_dicts(self.n_gens)

        self.gen_keys = []
        for gen in range(self.n_gens+1):
            self.gen_keys.append(gen)
        self.metric_dicts.add_keys_to_metrics(self.gen_keys)    

    def run(self):
        self.eval_env = texas_holdem.env(self.obs_type, render_mode='rgb_array')
   

        self.models = {}
        # create dict of pretrained models
        for i in range (0, 11):
            self.models['m' + str(i)] = PPO('MultiInputPolicy', self.eval_env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [256], 'vf': [256]},learning_rate= 0.005778633008004902, n_steps = 3072,  batch_size = 32, n_epochs= 70, ent_coef=  0.0025, vf_coef=  0.25, clip_range=0.1, max_grad_norm=0.6, gae_lambda = 0.85, normalize_advantage=False)
        # init learnt parameters to the models
        for key in self.models.keys():
            path_agent = os.path.join('S:\MSC_proj\ms', key)
            self.models[key].set_parameters(load_path_or_dict= path_agent)
        
        self.keys =list(self.models.keys()) 

       
        for i in range(1, len(self.keys)-1):
            mi = self.models[self.keys[i]]
            mi_p1 = self.models[self.keys[i+1]]

            self.eval_env.AGENT.policy = 'PPO'
            self.eval_env.AGENT.model = mi 

            self.eval_env.OPPONENT.policy = 'PPO'
            self.eval_env.OPPONENT.model = mi_p1

            ci = card_injector(self.eval_env.AGENT, self.eval_env.OPPONENT, self.eval_env)
            kl_div = ci.return_results()
            # self.results[str(i) + 'vs' +str(i+1)] = kl_div
            self.metric_dicts.update_sims(i, kl_div)
        gm = graph_metrics(n_models = 2, storage = self.metric_dicts, storageB= None, figsize=(6,8), t_steps = None, overlay=True, e_steps= None, title = 'KL divergence', device = 'pc')
        gm.print_select_graphs(False, False, False,False, False, True, False, False, False, False)
        return self.results