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

from self_play import extract_opt_act
class PPO_vs_OPPONENT():
    """
    Class to compare a PPO agent against different opponents in a Texas Hold'em environment.

    This class allows you to compare the performance of a PPO agent against different types of opponents
    (e.g., PPO, A2C) in a Texas Hold'em environment. It initializes the environment, loads model parameters,
    evaluates the agents, and retrieves results for analysis.

    Args:

    - obs_type (str): Observation type for the environment.
    - op_type (str): Type of opponent to compare against ('PPO' or 'A2C').

    Attributes:
    - type (str): Type of opponent ('PPO' or 'A2C').
    - obs_type (str): Observation type for the environment.
    - n_gens (int): Number of generations.
    - storageA (metric_dicts): Metric storage for the agent.
    - storageB (metric_dicts): Metric storage for the opponent.

    Methods:
    - init_eval_env(): Initialize the evaluation environment and agents.
    - load_params_from_file(path_agent, path_opponent): Load model parameters from files.
    - load_metric_dicts_storage(): Load metric dictionaries for storage.
    - evaluate(n_eval_episodes): Evaluate the agent and opponent in the environment.
    - get_results(graphs): Get and optionally plot experiment results.
    """
    def __init__(self, obs_type, op_type):
        
        self.type = op_type
        self.obs_type = obs_type
        self.n_gens = 1
        
        self.storageA= metric_dicts(self.n_gens)
        self.storageB= metric_dicts(self.n_gens)
       
    def init_eval_env(self):
        """
        Initialize the evaluation environment and agents.

        This method sets up the evaluation environment and initializes the agent and opponent models.
        """
        self.eval_env = texas_holdem.env(self.obs_type, render_mode='rgb_array')
        self.eval_env = Monitor(self.eval_env)
        self.eval_env.AGENT.policy = 'PPO'
        self.eval_env.OPPONENT.policy = self.type 
       
        self.eval_env.AGENT.model = PPO('MultiInputPolicy', self.eval_env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [256], 'vf': [256]},learning_rate= 0.005778633008004902, n_steps = 3072,  batch_size = 32, n_epochs= 70, ent_coef=  0.0025, vf_coef=  0.25)
        if self.eval_env.OPPONENT.policy == 'A2C':
            self.eval_env.OPPONENT.model = A2C('MultiInputPolicy', self.eval_env, optimizer_class = th.optim.Adam, activation_fn = nn.ReLU, net_arch=None, use_rms_prop = False)
        
        if self.eval_env.OPPONENT.policy == 'PPO':
            self.eval_env.OPPONENT.model =  PPO('MultiInputPolicy', self.eval_env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [256], 'vf': [256]},learning_rate= 0.005778633008004902, n_steps = 3072,  batch_size = 32, n_epochs= 70, ent_coef=  0.0025, vf_coef=  0.25, clip_range=0.1, max_grad_norm=0.6, gae_lambda = 0.85, normalize_advantage=False)
    
    def load_params_from_file(self, path_agent, path_opponent):
        """
        Load model parameters from files.

        Args:
        - path_agent (str): Path to the agent's model parameters file.
        - path_opponent (str): Path to the opponent's model parameters file.
        """
        self.eval_env.AGENT.model.set_parameters(load_path_or_dict= path_agent)
        if self.type == 'A2C' or self.type == 'PPO':
            self.eval_env.OPPONENT.model.set_parameters(load_path_or_dict= path_opponent)

    def load_metric_dicts_storage(self):
        """
        Load metric dictionaries for storage. StorageA is for agents and opponents, StorageB is for the random agent.
        """
        if self.eval_env.AGENT.policy ==self.eval_env.OPPONENT.policy:
            self.id_keys = [self.eval_env.AGENT.policy, self.eval_env.OPPONENT.policy + '_opponent']
        else:
            self.id_keys = [self.eval_env.AGENT.policy, self.eval_env.OPPONENT.policy]
        self.storageA.add_keys_to_metrics(self.id_keys)
        self.storageB.add_keys_to_metrics(self.id_keys)
    
    def evaluate(self, n_eval_episodes):
        """
        Evaluate the agent and opponent in the environment.

        Args:
        - n_eval_episodes (int): Number of evaluation episodes.
        """
        self.metric_dicts = self.storageA
        self.metric_dicts_rand = self.storageB
        self.n_eval_episodes = n_eval_episodes
        
        #evaluate agent and opponent
        print("evaluating agent and opponent")
        mean_reward ,episode_rewards, episode_lengths, episode_rewards_op= evaluate_policy(self.eval_env.AGENT.model, self.eval_env, callback=None, return_episode_rewards =True, n_eval_episodes = self.n_eval_episodes)

        # each episode rewards are exactly zero sum since the agent and opponent rewards are extracted from the same games 
        self.metric_dicts.gen_eval_rewards[ self.id_keys[0]] = episode_rewards
        self.metric_dicts.gen_eval_rewards[ self.id_keys[1]] = episode_rewards_op
        
        percentages_ag, percentages_op = extract_opt_act(self.eval_env)
        self.metric_dicts.update_eval_metrics_from_ep_rewards( self.id_keys[0], mean_reward,episode_rewards, percentages_ag, percentages_op)
        self.metric_dicts.update_eval_metrics_from_ep_rewards( self.id_keys[1], mean_reward,episode_rewards_op, percentages_ag, percentages_op)
        self.mean_reward = mean_reward
        self.eval_env.reset()
        
    def get_results(self, graphs):
        """
        Get and optionally plot experiment results.

        Args:
        - graphs (bool): Whether to plot graphs of experiment results.
        """
        print(self.metric_dicts.gen_eval_final_mean_reward)
        
        if graphs:
            gm = graph_metrics(n_models = 2, storage = self.metric_dicts, storageB= self.metric_dicts_rand, figsize=(6,8), t_steps = self.n_eval_episodes, overlay=True, e_steps= self.n_eval_episodes, title = str(self.eval_env.AGENT.policy) + '_vs_' + str(self.eval_env.OPPONENT.policy), device = 'pc')
            gm.print_all_graphs(False, True, False,False, True, True)