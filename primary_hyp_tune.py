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

class BatchMultiplier:
    """
    This class allows you to calculate the LCM of a list of numbers, find the LCM of the provided batch sizes,
    and generate a list of integers that are multiples of the LCM of batch sizes.
    
    The purpose of this class is to ensure what ever batchsize is tested for in the Optuna trial, a valid corresponding 
    value for number of steps is used. As such, the 'hyperparameter search' class inherits this class. 

    Attributes:
        batch_sizes (list): A list of batch sizes to be used for LCM calculation.

    Methods:
        gcd(a, b): Calculate the greatest common divisor (GCD) using the Euclidean algorithm.
        lcm(a, b): Calculate the least common multiple (LCM) of two numbers using the formula LCM(a, b) = (a * b) / GCD(a, b).
        lcm_of_list(numbers): Calculate the LCM of a list of numbers.
        generate_divisible_integers(): Generate a list of integers that are multiples of the LCM of batch sizes.
    """
    def __init__(self, batch_sizes):
        """
        Initialize the BatchMultiplier with a list of batch sizes.

        Args:
            batch_sizes (list): A list of positive integers representing batch sizes.
        """
        self.batch_sizes = batch_sizes[0]

    def gcd(self, a, b):
        """
        Calculate the greatest common divisor (GCD) of two integers using the Euclidean algorithm.

        Args:
            a (int): The first integer.
            b (int): The second integer.

        Returns:
            int: The GCD of the two input integers.
        """
        while b != 0:
            a, b = b, a % b
        return a

    def lcm(self, a, b):
        """
        Calculate the least common multiple (LCM) of two integers using the formula LCM(a, b) = (a * b) / GCD(a, b).

        Args:
            a (int): The first integer.
            b (int): The second integer.

        Returns:
            int: The LCM of the two input integers.
        """
        return (a * b) // self.gcd(a, b)

    def lcm_of_list(self, numbers):
        """
        Calculate the least common multiple (LCM) of a list of integers.

        Args:
            numbers (list): A list of positive integers.

        Returns:
            int: The LCM of the input list of integers.
        """
        result = 1
        for num in numbers:
            result = self.lcm(result, num)
        return result

    def generate_divisible_integers(self):
        """
        Generate a list of integers that are multiples of the LCM of batch sizes.

        Returns:
            list: A list of integers that are multiples of the LCM of batch sizes.
        """
        # Find the LCM of the batch sizes
        lcm_batch_sizes = self.lcm_of_list(self.batch_sizes)

        # Generate a list of integers that are multiples of the LCM
        divisible_integers = [i * lcm_batch_sizes for i in range(1, 6)]

        return divisible_integers

class hyperparam_search(BatchMultiplier):
    """
    Perform primary hyperparameter search.
    It allows optimizing hyperparameters for PPO and A2C models and
    provides a custom exception handler for robustness. The optimization is conducted
    using Optuna, and the best trial and parameters are stored for analysis.

    Args:
    - callback: A custom callback function for handling exceptions (optional).
    - verbose (bool): Whether to print verbose output during optimization.
    - batch_size (list): List of batch sizes to explore during optimization.
    - model_type (str): Reinforcement learning model type ('PPO' or 'A2C').
    - override_best (bool): Whether to override the best trial's parameters.
    - obs_type (str): Type of observation space in the game environment.

    Attributes:
    - callback: A custom callback function for handling exceptions.
    - verbose (bool): Whether verbose output is enabled.
    - batch_size (list): List of batch sizes to explore during optimization.
    - env: A reinforcement learning environment.
    - model_type (str): Reinforcement learning model type ('PPO' or 'A2C').
    - override_best (bool): Whether to override the best trial's parameters.
    - obs_type (str): Type of observation space in the game environment.
    - net_arch (list): List of neural network architectures to explore.
    - best_trial: Information about the best trial after optimization.
    - best_params: The best hyperparameters found during optimization.

    Methods:
    - init_trained_op(self): Initialize trained opponent models for evaluation.
    - optimize_ppo(self, trial): Optimize PPO model hyperparameters.
    - optimize_A2C(self, trial): Optimize A2C model hyperparameters.
    - custom_exception_handler(self, func): A custom exception handler decorator.
    - optimize_agent(self, trial): Optimize the agent's hyperparameters.
    - run(self, print_graphs): Run the hyperparameter optimization, store results and print graphs
    """

    def __init__(self, callback, verbose, batch_size, model_type, obs_type):
        self.callback = callback
        self.verbose = verbose
        self.batch_size = batch_size,
        self.env =None
        self.model_type = model_type
        self.obs_type = obs_type
        self.net_arch = [{'pi':[64,64], 'vf': [64,64]}, {'pi':[256], 'vf': [256]}, {'pi':[256,128], 'vf': [256,128]}]
        super().__init__(self.batch_size)

    
    def init_trained_op(self):
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
        na = self.net_arch
        for na_key in na:
            sp = self_play(0, 2048, 1, obs_type = self.obs_type, tag = 44, model = self.model_type, na_key = na_key, default_params=True, info = 'primhyptune')
            sp.run(False)
            self.na_gen_0_dict[str(na_key)] = sp.gen_lib[0]
            
    def optimize_ppo(self, trial):
            """

            This method defines a set of hyperparameters for tuning the PPO model using Optuna's
            suggestion methods. It specifies the hyperparameter search space for batch size, 
            number of training steps, learning rate, number of epochs, entropy coefficient, 
            value function coefficient, activation function, optimizer class, and neural network 
            architecture.

            Args:
            - trial: An Optuna trial object used for hyperparameter optimization.

            Returns:
            - dict: A dictionary containing the suggested hyperparameters for the PPO model.
            """
            return {
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512, 1024]),    
            'n_steps': trial.suggest_categorical('n_steps', self.generate_divisible_integers()), 
            'learning_rate': trial.suggest_loguniform('learning_rate',1e-6, 1),
            'n_epochs': int(trial.suggest_discrete_uniform('n_epochs', 10, 100, 10)),
            'ent_coef': int(trial.suggest_categorical('ent_coef', [0.000125, 0.0025, 0.005, 0.01, 0.02, 0.03])),
            'vf_coef': trial.suggest_categorical('vf_coef', [0.25, 0.5, 0.75, 0.80, 0.85]),
            'activation_fn': trial.suggest_categorical('activation_fn', [nn.ReLU, nn.Tanh]),
            'optimizer_class': trial.suggest_categorical('optimizer_class', [th.optim.Adam, th.optim.SGD]),
            'net_arch': trial.suggest_categorical('net_arch', [{'pi':[64,64], 'vf': [64,64]}, {'pi':[256], 'vf': [256]}, {'pi':[256,128], 'vf': [256,128]}])
            }
        
    def optimize_A2C(self, trial):
        """
        performs hyperparameter optimization specifically for the A2C agent model.
        It suggests values for various hyperparameters, including activation function, network
        architecture, learning rate, number of steps, entropy coefficient, value function coefficient,
        usage of RMSprop optimizer, and optimizer class.

        Args:
        - trial: An Optuna trial object used for hyperparameter optimization.

        Returns:
        - dict: A dictionary containing the optimized hyperparameters for the A2C agent.
        """    
        params = {
                'activation_fn': trial.suggest_categorical('activation_fn', [nn.ReLU, nn.Tanh]),
                'net_arch': trial.suggest_categorical('net_arch', [{'pi':[64,64], 'vf': [64,64]}, {'pi':[256], 'vf': [256]}, {'pi':[256,128], 'vf': [256,128]}]),
                'learning_rate': trial.suggest_loguniform('learning_rate',1e-6, 1),
                'n_steps': trial.suggest_categorical('n_steps', [1000, 2000, 5000, 10000]),
                'ent_coef': int(trial.suggest_categorical('ent_coef', [0.000125, 0.0025, 0.005, 0.01, 0.02, 0.03])),
                'vf_coef': trial.suggest_categorical('vf_coef', [0.25, 0.5, 0.75, 0.80, 0.85]),
                'use_rms_prop': trial.suggest_categorical('use_rms_prop', [True, False]),
                'optimizer_class': trial.suggest_categorical('optimizer_class', [th.optim.RMSprop])
                } 
        # If not using RMSprop, suggest an optimizer class from Adam or SGD
        if params['use_rms_prop'] == False:
                    params['optimizer_class']: trial.suggest_categorical('optimizer_class', [th.optim.Adam, th.optim.SGD])
                    
        return params
    
    def custom_exception_handler(self,func):
        """
        Custom exception handler decorator for handling exceptions in a function.

        This method is a decorator used to wrap another function. It captures and prints
        any exceptions that occur during the execution of the wrapped function and returns
        `None` in case of an exception. This decorator prevents it from crashing due to unhandled exceptions.

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
        Optimize agent hyperparameters and evaluate performance.

        This method performs hyperparameter optimization for the specified reinforcement learning
        agent model (PPO or A2C). It initializes the environment, policies, and hyperparameters
        based on the chosen model type. It then trains the agent using Proximal Policy Optimization
        (PPO) or Advantage Actor-Critic (A2C) and evaluates its performance.

        Args:
        - trial: An Optuna trial object used for hyperparameter optimization.

        Returns:
        - float: The mean reward achieved by the agent during evaluation.
        """
        callback = self.callback
        
       # initalize params depending on model
        if self.model_type == 'PPO':
            model_params = self.optimize_ppo(trial)
        if self.model_type == 'A2C':    
            model_params = self.optimize_A2C(trial) 
        #Initialize the training environment  
        env = texas_holdem.env(self.obs_type, render_mode="rgb_array")
        #Initialize the training environment
        Eval_env = texas_holdem.env(self.obs_type, render_mode="rgb_array")
        
         #Initialize the training environment  
        if self.model_type == 'PPO':
            env.AGENT.policy = 'PPO'
            env.OPPONENT.policy = 'PPO'
            Eval_env.AGENT.policy = 'PPO'
            Eval_env.OPPONENT.policy = 'random'
            
        elif self.model_type == 'A2C':
            env.AGENT.policy = 'A2C'
            env.OPPONENT.policy = 'A2C'
            Eval_env.AGENT.policy = 'A2C'
            Eval_env.OPPONENT.policy = 'random'
    
        # Initialize agent models and set opponent parameters
        if self.model_type == 'PPO':
            env.AGENT.model = PPO('MultiInputPolicy', env, verbose=0, **model_params)
            env.OPPONENT.model = PPO('MultiInputPolicy', env, verbose=0, **model_params)
            env.OPPONENT.model.set_parameters(self.na_gen_0_dict[str(model_params['net_arch'])])
            
            
        if self.model_type == 'A2C':
            env.AGENT.model = A2C('MultiInputPolicy', self.env, verbose=0, **model_params) 
            env.OPPONENT.model = A2C('MultiInputPolicy', self.env, verbose=0, **model_params)
            env.OPPONENT.model.set_parameters(self.na_gen_0_dict[str(model_params['net_arch'])])
            
            Eval_env.OPPONENT.model =  A2C('MultiInputPolicy', env, verbose=0, **model_params)
            Eval_env.OPPONENT.model.set_parameters(self.na_gen_0_dict[str(model_params['net_arch'])])
            
     # Initialize the evaluation environment with monitoring           
        Eval_env = Monitor(Eval_env)
        self.env = env
        cb = StopTrainingOnNoModelImprovement(1000, 75)
        # Set up evaluation callback    
        cb = EvalCallback(Eval_env, eval_freq=10000, callback_after_eval= cb, verbose=0, n_eval_episodes = 1000)
        try: 
            # Train the agent and evaluate its performance
            env.AGENT.model.learn(total_timesteps=40000, callback= cb, progress_bar=False, dumb_mode=False)
            mean_reward, _ = evaluate_policy(self.env.AGENT.model, Eval_env, n_eval_episodes=1000)
        except ValueError:
             # Handle the case where training fails with a ValueError
            mean_reward = 0.0
        return mean_reward
    
    def run(self, print_graphs):
        """
        Run the hyperparameter optimization process for the specified reinforcement learning model.

        This method performs hyperparameter optimization using Optuna for the specified reinforcement
        learning model (PPO or A2C). It optimizes hyperparameters such as batch size, learning rate,
        number of steps, and network architecture. The best trial and hyperparameters are returned.

        Args:
        - print_graphs (bool): If True, display optimization graphs after the optimization process.

        Returns:
        - Tuple: A tuple containing the best trial (Optuna trial object) and the best hyperparameters
        (dictionary) found during the optimization.

        Raises:
        - KeyboardInterrupt: If the optimization process is interrupted by the user, it will display
        optimization graphs before exiting.
        """
        # Apply custom exception handler to optimize_agent method
        self.optimize_agent = self.custom_exception_handler(self.optimize_agent)

        # Create an Optuna study for maximizing the objective function
        study = optuna.create_study(direction='maximize')

        try:
            # Run the hyperparameter optimization
            study.optimize(self.optimize_agent, callbacks=None, n_trials=2, n_jobs=1, show_progress_bar=True)
        except KeyboardInterrupt:
            print('Interrupted by keyboard.')
            # Display optimization graphs before exiting in case of interruption
            optuna.visualization.plot_param_importances(study).show()
            optuna.visualization.plot_optimization_history(study).show()

        # Store the best trial and best hyperparameters
        self.best_trial = study.best_trial
        self.best_params = study.best_params

        if print_graphs:
            # Display optimization graphs if requested
            optuna.visualization.plot_param_importances(study).show()
            optuna.visualization.plot_optimization_history(study).show()


def primary_hyp_search_PPO():
    """
    Perform hyperparameter search for PPO model and print results.

    This function initializes a hyperparameter search instance with specified parameters,
    conducts a hyperparameter search experiment, and prints the results including graphs.

    Args:
    - None

    Returns:
    - None
    """
    hypsrch = hyperparam_search(callback=None, verbose=True, batch_size=[32, 64, 128, 256, 512, 1024], model_type='PPO', obs_type='72+')
    hypsrch.init_trained_op()
    print(hypsrch.run(print_graphs=True))

primary_hyp_search_PPO()

def primary_hyp_search_A2C():
    """
    Perform hyperparameter search experiments for an A2C model and print results.

    This function initializes a hyperparameter search instance with specified parameters,
    conducts a hyperparameter search experiment, and prints the results including graphs.

    Args:
    - None

    Returns:
    - None
    """
    hypsrch = hyperparam_search(callback=None, verbose=True, batch_size=[32, 64, 128, 256, 512, 1024], model_type='A2C', obs_type='72+')
    hypsrch.init_trained_op()
    print(hypsrch.run(print_graphs=True))

primary_hyp_search_A2C()