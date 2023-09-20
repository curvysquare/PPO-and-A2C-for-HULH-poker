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
from self_play import self_play
from self_play import CustomLoggerCallback

class obs_experiment():
    """
    Observation Experiment class for training and evaluating agents in different environments.

    Args:
    - total_timesteps (int): Total training timesteps.
    - n_eval_episodes (int): Number of episodes for evaluation.
    - model (str): Model type for the agents.

    This class performs training and evaluation experiments on reinforcement learning agents in various environments.
    It allows for training multiple agents and evaluating them across different environments.

    Methods:
    - train_opponents(gen_to_load): Train opponent agents for each environment and store the trained models.
    - init_agent_opponent_models(): Initialize the agent and opponent models for training and evaluation environments.
    - agent_train_eval(): Train agents and evaluate their performance on both training and evaluation environments.
    - get_results(graphs): Get experiment results, including final mean rewards, and optionally plot graphs.

    The class uses custom metrics and callbacks to collect and store training and evaluation results.
    """
    def __init__(self, total_timesteps, n_eval_episodes, model):
        self.total_timesteps = total_timesteps
        self.n_eval_episodes = n_eval_episodes
        self.model = model
        # self.envs = obs_type_envs()
        self.train_envs = {'124': texas_holdem.env('124', render_mode='rgb_array') , '72+': texas_holdem.env('72+', render_mode='rgb_array') , '72': texas_holdem.env('72', render_mode='rgb_array') }
        self.eval_envs = {'124': Monitor(texas_holdem.env('124', render_mode='rgb_array')) , '72+': Monitor(texas_holdem.env('72+', render_mode='rgb_array')) , '72': Monitor(texas_holdem.env('72', render_mode='rgb_array')) }
   
        self.train_envs_keys = list(self.train_envs.keys())
        self.eval_envs_keys = list(self.eval_envs.keys())

        self.metric_dicts = metric_dicts(3)
        self.metric_dicts.add_keys_to_metrics_dict(self.train_envs, self.eval_envs)
        
        self.metric_dicts_rand = metric_dicts(3)
        self.metric_dicts_rand.add_keys_to_metrics_dict(self.train_envs, self.eval_envs)

        self.na = {'pi': [64], 'vf': [64]}

    def train_opponents(self, gen_to_load):
        """
        Train opponent agents for each environment and store the trained models.

        Args:
        - gen_to_load (int): Generation of opponent agents to load and train against.

        This method initializes self-play environments, trains opponent agents, and stores the trained models.
        """
        self.trained_opponents = {}
        for key in self.train_envs_keys:
            sp = self_play(0, 2048, 1, obs_type = key, tag = key, model = self.model, na_key = self.na, default_params = True, info = 'obs_exp')
            sp.run(False)
            root = sp.gen_lib[gen_to_load]
            self.trained_opponents[key] = root
 
        
    def init_agent_opponent_models(self):
        """
        Initialize agent and opponent models for training and evaluation environments.

        This method initializes agent and opponent models for both training and evaluation environments.
        It sets policies, network architectures, and loads trained opponent models for the environments.
        """
        # Initialize models for training environments
        for id in self.train_envs_keys:
            env = self.train_envs[id]
            env.OPPONENT.policy = 'PPO'
            env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
            env.AGENT.policy = 'PPO'
            env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
            env.OPPONENT.model.set_parameters(load_path_or_dict= self.trained_opponents[id])
        # Initialize models for evaluation environments    
        for id in self.eval_envs_keys:
            env = self.eval_envs[id]
            env.OPPONENT.policy = 'PPO'
            env.AGENT.policy = 'PPO'
            env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
            env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
            env.OPPONENT.model.set_parameters(self.trained_opponents[id])
     
    def agent_train_eval(self):
        """
        Train agents and evaluate their performance on training and evaluation environments.

        This method trains agents using the specified total training timesteps and evaluates their performance.
        It collects training and evaluation metrics, such as rewards and losses.
        """
        for id in self.train_envs_keys:  
            callback_train = CustomLoggerCallback()     
            env = self.train_envs[id]
            print(id, "learning")
            env.AGENT.model.learn(self.total_timesteps, dumb_mode = False,progress_bar=True, callback=callback_train)
            self.metric_dicts.update_train_metrics_from_callback(id, callback_train)
            rews = callback_train.rewards
            self.metric_dicts.update_train_metrics_from_callback(id, callback_train) 
                
            env.reset()
                
            callback_train_rand_op = CustomLoggerCallback()              
            env.AGENT.model.learn(total_timesteps = self.total_timesteps, dumb_mode = True, callback=callback_train_rand_op , progress_bar=True,)
            self.metric_dicts_rand.update_train_metrics_from_callback(id, callback_train_rand_op)
 
        for id in self.eval_envs_keys:
            env = self.eval_envs[id]
            env.AGENT.model = self.train_envs[id].AGENT.model
            mean_reward,episode_rewards, episode_lengths= evaluate_policy(env.AGENT.model, env, callback=None, return_episode_rewards =True, n_eval_episodes = self.n_eval_episodes)
            self.metric_dicts.update_eval_metrics_from_ep_rewards(id, mean_reward,episode_rewards)
            callback_eval_rand_op = CustomLoggerCallback()
            env.AGENT.model.learn(total_timesteps = self.n_eval_episodes, dumb_mode = True, callback=callback_eval_rand_op , progress_bar=True,)
            mean_reward = callback_eval_rand_op.final_mean_reward
            episode_rewards = callback_eval_rand_op.rewards
            self.metric_dicts_rand.update_eval_metrics_from_ep_rewards(gen = id, mean_reward = mean_reward, episode_rewards = episode_rewards)

    def get_results(self, graphs):
        """
        Get experiment results and optionally plot graphs.

        Args:
        - graphs (bool): If True, plot graphs; otherwise, skip plotting.

        Returns:
        - list: A list containing the final mean reward for training and evaluation.

        This method retrieves the final mean rewards for both training and evaluation environments.
        If the 'graphs' argument is True, it also plots various performance-related graphs using the provided metrics.

        Returns a list with two elements: the final mean reward for training and the final mean reward for evaluation.
        """
        print(self.metric_dicts.gen_train_final_mean_reward)
        print(self.metric_dicts.gen_eval_final_mean_reward)
        
        if graphs:
            gm = graph_metrics(n_models = 3, storage = self.metric_dicts, storageB= self.metric_dicts_rand, figsize= (10, 8), t_steps = self.total_timesteps)
            gm.create_x_y()
            gm.plot_rewards(True, True)
            gm.plot_moving_rewards(True, True)
            gm.plot_moving_mean(True, True)
            gm.plot_loss()
        
        return [self.metric_dicts.gen_train_final_mean_reward, self.metric_dicts.gen_eval_final_mean_reward]

def obs_experiment_group():
    """
    Run a group of observational experiments.

    This function sets up and executes a series of observational experiments using the `obs_experiment` class.
    It initializes experiments, trains opponents, initializes agent and opponent models, runs training and evaluation,
    and retrieves and optionally plots experiment results. 
    
    Returns:
    - None
    """
    # n_eval_episodes will always be at least n_epochs because running a dumb agent.

    # Create an observational experiment
    # experiment = obs_experiment(total_timesteps=35000, n_eval_episodes=2048, model='PPO')
    experiment = obs_experiment(total_timesteps=2048, n_eval_episodes=2048, model='PPO')

    # Train opponents for the experiment
    experiment.train_opponents(gen_to_load=0)

    # Initialize agent and opponent models
    experiment.init_agent_opponent_models()

    # Run training and evaluation for the experiment
    print(experiment.agent_train_eval())

    # Get and optionally plot experiment results
    experiment.get_results(graphs=True)

obs_experiment_group() 