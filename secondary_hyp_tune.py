import numpy as np   
import texas_holdem_mod as texas_holdem
from rlcard.utils.utils import print_card as prnt_cd
from rlcard.utils.utils import print_card as prnt_cd
from env_checker_mod import check_env
from evaluation_mod import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
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
from self_play import self_play

class micro_hyperparam_search:
    """
    A class for performing micro-level hyperparameter search for reinforcement learning models.

    This class facilitates hyperparameter optimization for reinforcement learning models,
    specifically PPO (Proximal Policy Optimization) and A2C (Advantage Actor-Critic).
    It allows you to fine-tune hyperparameters such as learning rate, batch size, and network
    architecture using Optuna, a hyperparameter optimization library.

    Args:
    - callback: A custom callback function for early stopping or other purposes.
    - verbose: A boolean indicating whether to display verbose output.
    - model_type: The type of RL model to optimize ('PPO' or 'A2C').
    - override_best: A boolean indicating whether to override previous best hyperparameters.
    - obs_type: The observation type used in the RL environment.

    Methods:
    - init_trained_op(): Initialize a pre-trained RL agent for hyperparameter optimization.
    - optimize_ppo(trial): Optimize hyperparameters for the PPO model using Optuna.
    - optimize_A2C(trial): Optimize hyperparameters for the A2C model using Optuna.
    - custom_exception_handler(func): A decorator to handle exceptions gracefully.
    - optimize_agent(trial): Perform hyperparameter optimization for the specified RL model.
    - run(print_graphs): Run the hyperparameter optimization process and return the best trial
      and hyperparameters found.

    Attributes:
    - callback: A custom callback function for early stopping or other purposes.
    - verbose: A boolean indicating whether to display verbose output.
    - env: The RL environment used for training and evaluation.
    - model_type: The type of RL model to optimize ('PPO' or 'A2C').
    - override_best: A boolean indicating whether to override previous best hyperparameters.
    - obs_type: The observation type used in the RL environment.
    - best_trial: The best trial object found during hyperparameter optimization.
    - best_params: The best hyperparameters found during hyperparameter optimization.
    """
    def __init__(self, callback, verbose, model_type, obs_type):
        self.callback = callback
        self.verbose = verbose
        self.env =None
        self.model_type = model_type
        self.obs_type = obs_type

        
    def init_trained_op(self, training_steps):
        """
        Initialize trained opponent models for evaluation. the trained opponent has to have the same
        network architecture as the agent. This function creates a dictionary according to this    

        This method initializes trained opponent models for each specified neural network architecture
        (na_key) as generation zero It uses the self_play class to create and run
        self-play experiments for each architecture, saving the model for the opponent
        at generation zero.

        Args:
        - None

        Returns:
        - None
        """
       
        self.na_gen_0_dict = {}
        na_key = {'pi': [256], 'vf': [256]}
        sp = self_play(0, training_steps, 1, obs_type = self.obs_type, tag = 202, model = self.model_type, na_key = na_key, default_params=False, info='secondhyptune'+str(self.model_type))
        sp.run(False)
        self.na_gen_0_dict[str(na_key)] = sp.gen_lib[0]
            
        self.root = sp.gen_lib[0]
        
    def optimize_ppo(self, trial):
        """
        Optimize hyperparameters for the Proximal Policy Optimization (PPO) model using Optuna.

        Args:
        - trial: An Optuna Trial object used for hyperparameter optimization.

        Returns:
        - dict: A dictionary containing the optimized hyperparameters for the PPO model.
          The dictionary includes the following keys:
          - 'gae_lambda': The generalized advantage estimation lambda parameter.
          - 'clip_range': The clipping range for PPO loss.
          - 'normalize_advantage': Whether to normalize advantages.
          - 'max_grad_norm': The maximum gradient norm for gradient clipping.
          - 'net_arch': The neural network architecture configuration for policy and value functions.
        """
        return {
            'gae_lambda': trial.suggest_categorical('gae_lambda', [0.85, 0.90, 0.95]),
            'clip_range': trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3]),
            'normalize_advantage': trial.suggest_categorical('normalize_advantage', [True, False]),
            'max_grad_norm': trial.suggest_categorical('max_grad_norm', [0.4, 0.5, 0.6]),
            'net_arch': trial.suggest_categorical('net_arch', [{'pi': [256], 'vf': [256]}])
        }

    def optimize_A2C(self, trial):
        """
        Optimize hyperparameters for the Advantage Actor-Critic (A2C) model using Optuna.

        Args:
        - trial: An Optuna Trial object used for hyperparameter optimization.

        Returns:
        - dict: A dictionary containing the optimized hyperparameters for the A2C model.
          The dictionary includes the following keys:
          - 'gae_lambda': The generalized advantage estimation lambda parameter.
          - 'normalize_advantage': Whether to normalize advantages.
          - 'max_grad_norm': The maximum gradient norm for gradient clipping.
          - 'net_arch': The neural network architecture configuration for policy and value functions.
        """
        return {
            'gae_lambda': trial.suggest_categorical('gae_lambda', [0.85, 0.90, 0.95]),
            'normalize_advantage': trial.suggest_categorical('normalize_advantage', [True, False]),
            'max_grad_norm': trial.suggest_categorical('max_grad_norm', [0.3, 0.4, 0.5, 0.6]),
            'net_arch': trial.suggest_categorical('net_arch', [{'pi': [256], 'vf': [256]}])
        }
       
    def custom_exception_handler(self,func):
        """
        Custom exception handler decorator for handling exceptions in a function.

        This method is a decorator used to wrap another function. It captures and prints
        any exceptions that occur during the execution of the wrapped function and returns
        `None` in case of an exception. This decorator helps improve the robustness of the
        wrapped function by preventing it from crashing due to unhandled exceptions.

        Args:
        - func: The function to be wrapped and protected from exceptions.

        Returns:
        - wrapper: A wrapped version of the input function with exception handling.
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Exception caught: {e}")
                return None
        return wrapper

    def optimize_agent(self, trial):
        """
        Optimize the reinforcement learning agent using the specified model type (PPO or A2C) and hyperparameters.

        Args:
        - trial: An Optuna Trial object for hyperparameter optimization.

        Returns:
        - float: The mean reward achieved by the optimized agent.

        This method initializes an environment and evaluation environment for Texas Hold'em.
        The agent's policy and model are set based on the chosen model type (PPO or A2C).
        The agent's model is trained with the specified hyperparameters.
        The mean reward achieved by the agent is returned as the optimization objective.

        For PPO:
        - Uses the PPO algorithm with hyperparameters specified by 'model_params'.
        - 'model_params' should contain 'optimizer_class', 'activation_fn', 'learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'ent_coef', 'vf_coef', and 'net_arch'.

        For A2C:
        - Uses the A2C algorithm with hyperparameters specified by 'model_params'.
        - 'model_params' should contain 'optimizer_class', 'activation_fn', 'learning_rate', 'n_steps', 'ent_coef', 'vf_coef', 'use_rms_prop', and 'net_arch'.

        The training process is wrapped in a try-except block to handle exceptions and return a mean reward of 0 if training fails.
        """
        callback = self.callback
        
        # Initialize parameters depending on the model
        if self.model_type == 'PPO':
            model_params = self.optimize_ppo(trial)
        elif self.model_type == 'A2C':    
            model_params = self.optimize_A2C(trial) 
        
        # Initialize the environment
        env = texas_holdem.env(self.obs_type, render_mode="rgb_array")
        
        # Initialize the evaluation environment
        Eval_env = texas_holdem.env(self.obs_type, render_mode="rgb_array")
        
        # Initialize policies
        if self.model_type == 'PPO':
            env.AGENT.policy = 'PPO'
            env.OPPONENT.policy = 'PPO'
            Eval_env.AGENT.policy = 'PPO'

        elif self.model_type == 'A2C':
            env.AGENT.policy = 'A2C'
            env.OPPONENT.policy = 'A2C'
            Eval_env.AGENT.policy = 'A2C'
        
        if self.model_type == 'PPO':
            env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class=th.optim.Adam, 
                                activation_fn=nn.Tanh, learning_rate=0.005778633008004902, n_steps=3072, 
                                batch_size=32, n_epochs=70, ent_coef=0.0025, vf_coef=0.25, verbose=0, **model_params)
            env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class=th.optim.Adam, 
                                    activation_fn=nn.Tanh, learning_rate=0.005778633008004902, n_steps=3072, 
                                    batch_size=32, n_epochs=70, ent_coef=0.0025, vf_coef=0.25, verbose=0, **model_params)
            env.OPPONENT.model.set_parameters(self.na_gen_0_dict[str(model_params['net_arch'])])
        
        if self.model_type == 'A2C':
            env.AGENT.model = A2C('MultiInputPolicy', env, optimizer_class=th.optim.RMSprop, 
                                activation_fn=nn.ReLU, learning_rate=0.04568216636850521, n_steps=10000, 
                                ent_coef=0.0025, vf_coef=0.25, use_rms_prop=False, **model_params)
            env.OPPONENT.model = A2C('MultiInputPolicy', env, optimizer_class=th.optim.RMSprop, 
                                    activation_fn=nn.ReLU, learning_rate=0.04568216636850521, n_steps=10000, 
                                    ent_coef=0.0025, vf_coef=0.25, use_rms_prop=False, **model_params)
            env.OPPONENT.model.set_parameters(self.na_gen_0_dict[str(model_params['net_arch'])])
        
        Eval_env = Monitor(Eval_env)
        Eval_env.OPPONENT.policy = 'random'
    
        try: 
            env.AGENT.model.learn(total_timesteps=30720, callback=None, progress_bar=False, dumb_mode=False)
            mean_reward, _ = evaluate_policy(env.AGENT.model, Eval_env, n_eval_episodes=10000)
        except ValueError:
            mean_reward = 0.0
        return mean_reward
        # n trials = 20

    def run(self, print_graphs, numb_trials):
        self.optimize_agent = self.custom_exception_handler(self.optimize_agent)
        if __name__ == '__main__':
            study = optuna.create_study(direction= 'maximize')
        try:
            study.optimize(self.optimize_agent, callbacks=None, n_trials=20, n_jobs=1, show_progress_bar=True)
        except KeyboardInterrupt:
            print('Interrupted by keyboard.')
            optuna.visualization.plot_param_importances(study).show()
            optuna.visualization.plot_optimization_history(study).show()

        self.best_trial = study.best_trial
        self.best_params = study.best_params
        
        if print_graphs == True:
            optuna.visualization.plot_param_importances(study).show()
            optuna.visualization.plot_optimization_history(study).show()
        
        return self.best_trial, self.best_params
        
