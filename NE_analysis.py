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

class NE_tool():
    """
    A class for analyzing and visualizing Nash Equilibrium (NE) properties in a Texas Hold'em environment.

    This class initializes multiple pre-trained PPO models and evaluates their NE-related properties, including
    equilibrium rewards, best response rewards, and regrets. It also provides functionality to visualize the results.

    Parameters:
    - obs_type (str): The observation type for the Texas Hold'em environment ('rgb_array' or 'tensor').
    - n_gens (int): The number of generations to analyze.

    Attributes:
    - obs_type (str): The observation type for the environment.
    - eq_rew (dict): A dictionary to store equilibrium rewards for each model.
    - br_rew (dict): A dictionary to store best response rewards for each model.
    - regrets (dict): A dictionary to store regrets for each model.
    - n_gens (int): The number of generations to analyze.
    - metric_dicts: An instance of the metric_dicts class to store metric data.
    - gen_keys (list): A list of generation keys for tracking the analysis.

    Methods:
    - run(n_eval_episodes): Runs the analysis for equilibrium, best response, and regrets.
    - evaluate(modelA, modelB): Evaluates the mean reward of a pair of models in the Texas Hold'em environment.
    - print_graph(): Generates and displays a line plot of regrets across the self-play process.
    """
    def __init__(self, obs_type, n_gens):
        self.obs_type = obs_type
        self.eq_rew = {}
        self.br_rew = {}
        self.regrets = {}
        self.n_gens = n_gens
        self.metric_dicts = metric_dicts(self.n_gens)

        self.gen_keys = []
        for gen in range(self.n_gens+1):
            self.gen_keys.append(gen)
        self.metric_dicts.add_keys_to_metrics(self.gen_keys)

    def run(self,  n_eval_episodes):
        self. n_eval_episodes =  n_eval_episodes
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

        for i in range(2, len(self.keys)):
            print(i)
            mi = self.models[self.keys[i]]
            mi_p1 = self.models[self.keys[i-1]]
            mi_p2 = self.models[self.keys[i-2]]

            eq_reward_mi = self.evaluate(mi_p1, mi_p2)
            br_payoff_mi = self.evaluate(mi, mi_p1)
            regret_mi = br_payoff_mi- eq_reward_mi

            self.eq_rew[self.keys[i]] = eq_reward_mi
            self.br_rew[self.keys[i]] = br_payoff_mi
            self.regrets[self.keys[i]] = regret_mi

    def evaluate(self, modelA, modelB):
        self.eval_env = texas_holdem.env(self.obs_type, render_mode='rgb_array')
        self.eval_env = Monitor(self.eval_env)
        self.n_eval_episodes =  self.n_eval_episodes
        self.eval_env.AGENT.policy = 'PPO'
        self.eval_env.AGENT.model = modelA
        self.eval_env.OPPONENT.policy = 'PPO'
        self.eval_env.OPPONENT.model = modelB

        mean_reward ,episode_rewards, episode_lengths, episode_rewards_op= evaluate_policy(self.eval_env.AGENT.model, self.eval_env, callback=None, return_episode_rewards =True, n_eval_episodes = self.n_eval_episodes)

        return mean_reward


    def print_graph(self):
        x_values = list(self.regrets.keys())
        y_values = list(self.regrets.values())

        # Create a line plot
        plt.figure(figsize=(10, 6))  # Optional: set the figure size
        plt.plot(x_values, y_values, marker='o', linestyle='-')
        plt.title('regret across the selfplay process')
        plt.xlabel('selfplay model' )
        plt.ylabel('regret')
        plt.grid(True)

        # Show the plot
        plt.tight_layout()  
        plt.show()