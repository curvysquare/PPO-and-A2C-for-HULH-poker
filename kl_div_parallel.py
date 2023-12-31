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
from classmaker import obs_type_envs
from classmaker import metric_dicts

from injector import card_injector
from human_input import human_play


class kl_div_test():
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
   

        self.models_IIG = {}
        # create dict of pretrained models
        for i in range (0, 11):
            self.models_IIG['mIIG' + str(i)] = PPO('MultiInputPolicy', self.eval_env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [256], 'vf': [256]},learning_rate= 0.005778633008004902, n_steps = 3072,  batch_size = 32, n_epochs= 70, ent_coef=  0.0025, vf_coef=  0.25, clip_range=0.1, max_grad_norm=0.6, gae_lambda = 0.85, normalize_advantage=False)
        # init learnt parameters to the models
        for key in self.models_IIG.keys():
            path_agent = os.path.join('S:\MSC_proj\ms', key)
            self.models_IIG[key].set_parameters(load_path_or_dict= path_agent)

        self.keys_IIG =list(self.models_IIG.keys()) 

        self.models_PIG = {}
        # create dict of pretrained models
        for i in range (0, 11):
            self.models_PIG['mPIG' + str(i)] = PPO('MultiInputPolicy', self.eval_env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [256], 'vf': [256]},learning_rate= 0.005778633008004902, n_steps = 3072,  batch_size = 32, n_epochs= 70, ent_coef=  0.0025, vf_coef=  0.25, clip_range=0.1, max_grad_norm=0.6, gae_lambda = 0.85, normalize_advantage=False)
        # init learnt parameters to the models
        for key in self.models_PIG.keys():
            path_agent = os.path.join('S:\MSC_proj\ms', key)
            self.models_PIG[key].set_parameters(load_path_or_dict= path_agent)

        self.keys_PIG =list(self.models_PIG.keys()) 
       
        for i in range(1, len(self.keys_PIG)-1):
            mi = self.models_IIG[self.keys_IIG[i]]
            mi_2 = self.models_PIG[self.keys_PIG[i+1]]

            self.eval_env.AGENT.policy = 'PPO'
            self.eval_env.AGENT.model = mi 

            self.eval_env.OPPONENT.policy = 'PPO'
            self.eval_env.OPPONENT.model = mi_2

            ci = card_injector(self.eval_env.AGENT, self.eval_env.OPPONENT, self.eval_env)
            kl_div = ci.return_results()
            # self.results[str(i) + 'vs' +str(i+1)] = kl_div
            self.metric_dicts.update_sims(i, kl_div)
        gm = graph_metrics(n_models = 2, storage = self.metric_dicts, storageB= None, figsize=(6,8), t_steps = None, overlay=True, e_steps= None, title = 'KL divergence', device = 'pc')
        gm.print_select_graphs(False, False, False,False, False, True, False)    
        return self.results

kl = kl_div_test('72+',10)
print(kl.run())         