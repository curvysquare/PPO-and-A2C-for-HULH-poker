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

def extract_opt_act(env):
    """
    Extracts and computes the percentages of occurrences of the optimal action '1' for both the agent (acts_ag) and the opponent (acts_op) 
    from the given environment (env). The percentages are computed as the ratio of the number of '1's to the total number of 
    elements, expressed as percentages.

    Parameters:
    - env (Environment): The environment object containing the action sequences for the agent and opponent.

    Returns:
    - percentages_ag (list of float): A list of percentages representing the occurrences of '1', reprenting the optimal action was taken,  in the agent's actions.
    - percentages_op (list of float): A list of percentages representing the occurrences of '1', reprenting the optimal action was taken,  in the opponent's actions.

    Note:
    - If there are no elements in either acts_ag or acts_op, the corresponding percentage will be set to 0.0.
    - The percentages are computed incrementally for each action, starting from the 21st action so the plotted graph is smoother since the percentage has stabalised after the first 20 points.
    """
    acts_ag = env.opt_acts_ag
    acts_op = env.opt_acts_op
    
    # Initialize lists to store percentages
    percentages_ag = []
    percentages_op = []
    
    # Calculate percentages for agent
    total_elements_ag = len(acts_ag)
    ones_count_ag = acts_ag.count(1)
    
    if total_elements_ag == 0:
        percentages_ag.append(0.0)
    else:
        for i, act in enumerate(acts_ag[20:], start=21):
            ones_count_ag = acts_ag[:i+1].count(1)
            percentage_ag = (ones_count_ag / (i + 1)) * 100
            percentages_ag.append(percentage_ag)
    
    # Calculate percentages for opponent
    total_elements_op = len(acts_op)
    ones_count_op = acts_op.count(1)
    
    if total_elements_op == 0:
        percentages_op.append(0.0)
    else:
        for i, act in enumerate(acts_op[20:], start=21):
            ones_count_op = acts_op[:i+1].count(1)
            percentage_op = (ones_count_op / (i + 1)) * 100
            percentages_op.append(percentage_op)
    
    return percentages_ag, percentages_op

class CustomLoggerCallback(BaseCallback):
    """
    A custom callback for logging various training metrics during training.

    This callback keeps track of value losses, policy losses, entropy losses, rewards, and moving
    average rewards over episodes. It also records the cumulative sum of rewards over time and
    the number of steps taken in the training process.

    Attributes:
        value_losses (list): A list to store value losses during training.
        policy_losses (list): A list to store policy losses during training.
        entropy_losses (list): A list to store entropy losses during training.
        moving_mean_reward (list): A list to store the moving average of rewards over episodes.
        rewards (list): A list to store the rewards obtained in each episode.
        moving_total (list): A list to store the cumulative sum of rewards over episodes.
        step_list (list): A list to store the number of steps taken in the training process.
        opt_acts_over_eps (list): A list for future use or customization.

    Methods:
        _on_step(): This method is called after each training step. It records losses and updates
            reward-related metrics.

    Note:
        The final_mean_reward attribute is computed based on the moving mean reward and is available
        when there is more than one recorded moving mean reward value.
    """
    def __init__(self, verbose=0):
        super(CustomLoggerCallback, self).__init__(verbose)
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []

        self.moving_mean_reward = []
        self.rewards = []
        self.moving_total = [0]
        self.step_list = []
        self.opt_acts_over_eps = []

        if len(self.moving_mean_reward) > 1:
            self.final_mean_reward = self.moving_mean_reward[-1]

    def _on_step(self) -> bool:
        """
        Callback method called after each training step.

        It records value losses, policy losses, entropy losses, and updates reward-related metrics
        like rewards and moving average rewards.

        It checks if the required attribute is present since the loss values are only initialized after
        the first value network update, equivalent to the number of steps required to fill the replay buffer.

        Returns:
            bool: Always returns True.
        """
        if hasattr(self.model, 'value_loss'):
            if self.model.value_loss is not None:
                loss = self.model.value_loss.item()
                loss = float(loss)
                self.value_losses.append(loss)
        if hasattr(self.model, 'policy_loss'):
            if self.model.policy_loss is not None:
                loss = self.model.policy_loss.item()
                loss = float(loss)
                self.policy_losses.append(loss)
        if hasattr(self.model, 'entropy_loss'):
            if self.model.entropy_loss is not None:
                loss = self.model.entropy_loss.item()
                loss = float(loss)
                self.entropy_losses.append(loss)

        self.step_list.append(self.num_timesteps)
        self.rewards.append(self.model.env.buf_rews[0])
        self.moving_mean_reward.append(np.mean(self.rewards))
        self.moving_total.append(self.moving_total[-1] + self.model.env.buf_rews[0])
        if len(self.moving_mean_reward) > 1:
            self.final_mean_reward = self.moving_mean_reward[-1]

        return True
       
class self_play():
    """
    Self-play structure for training and evaluating agents, saving the 
    trained models and their corresponding metrics, and then plotting the results. 
    A random opponenent is evaluated in parallel to provide a benchmark.

    Args:
    - n_gens (int): Number of generations for training and evaluation.
    - learning_steps (int): Number of training steps per generation.
    - n_eval_episodes (int): Number of episodes for evaluation.
    - obs_type (str): Type of observation space in the game environment.
    - tag (str): A tag used for naming the selfplay.
    - model (str): Reinforcement learning model type ('PPO' or 'A2C').
    - na_key (str): Specification of the agents network architecture if required.
    - default_params (bool): Whether to use default hyperparameters for the model.
    - info (str): Additional information for the selplay training to be used in each models title for saving.

    Attributes:
    - n_gens (int): Number of generations.
    - learning_steps (int): Number of training steps per generation.
    - n_eval_episodes (int): Number of evaluation episodes.
    - obs_type (str): Type of observation space in the game environment.
    - tag (str): tag for saving purposes
    - model (str): Reinforcement learning model type ('PPO' or 'A2C').
    - n_steps (int): Number of steps in each training episode.
    - na_key (str): Specification of the agents network architecture if required.
    - default_params (bool): Whether default hyperparameters are used.
    - title (str): Title for the experiment based on parameters and info.
    - base_model: The base reinforcement learning model used for training and evaluation.
    - gen_lib (dict): A dictionary containing file paths for saving models after each generations training has completed.
    - gen_keys (list): A list of generation keys.
    - metric_dicts: A data storage object for tracking training and evaluation metrics.
    - metric_dicts_rand: A data storage object for tracking metrics of a random opponent.

    Methods:
    - create_files(self, n_files, device): Create folders and return file paths for saving the generation models.
    - run(self, eval_opponent_random): Run the self-play training and evaluation loop.
    - get_results(self, graphs): Generate and display graphs based on collected metrics.
    """
    def __init__(self, n_gens, learning_steps, n_eval_episodes, obs_type, tag, model, na_key, default_params, info): 
        self.n_gens = n_gens
        self.learning_steps = learning_steps
        self.n_eval_episodes = n_eval_episodes
        self.obs_type  = obs_type   
        self.tag = tag
        self.model = model
        self.n_steps = 3072
        self.na_key = na_key
        self.default_params = default_params
        
        self.title = model + obs_type + str(n_gens)+ 'default' + str(default_params) + info

        # initialize the agent model depending on if default hyperparameters are required.
        if self.model == 'PPO':
            if default_params:
                 self.base_model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [64], 'vf': [64]})
            else:      
                 self.base_model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [256], 'vf': [256]},learning_rate= 0.005778633008004902, n_steps =  self.n_steps,  batch_size = 32, n_epochs= 70, ent_coef=  0.0025, vf_coef=  0.25, clip_range=0.1, max_grad_norm=0.6, gae_lambda = 0.85, normalize_advantage=False)
            self.env.AGENT.policy = 'PPO'
        elif self.model == 'A2C':
            self.base_model = A2C('MultiInputPolicy', env, optimizer_class= th.optim.RMSprop, activation_fn = nn.ReLU, learning_rate =  0.04568216636850521, n_steps =10000, ent_coef =0.0025, vf_coef = 0.25, use_rms_prop = False,  net_arch = {'pi': [256], 'vf': [256]}, gae_lambda=0.85, normalize_advantage=False, max_grad_norm=0.5)
            self.env.AGENT.policy = 'A2C'
        
        # create files to save the models, returns a dictionary of the file paths.
        self.gen_lib = self.create_files(self.n_gens + 1, device='pc') 
        self.gen_keys = []
        for gen in range(n_gens+1):
            self.gen_keys.append(gen)
            
        # intiliase the data storage object for the agents and the benchmark random opponent
        # and add the required keys.
        self.metric_dicts = metric_dicts(self.n_gens)
        self.metric_dicts.add_keys_to_metrics(self.gen_keys)
        self.metric_dicts_rand = metric_dicts(self.n_gens)
        self.metric_dicts_rand.add_keys_to_metrics(self.gen_keys)
        
    def create_files(self, n_files, device):
        if device == 'pc':
            directory = r'S:/MSC_proj/models'
        elif device == 'mac':    
            directory = '/Users/rhyscooper/Desktop/MSc Project/Pages/models'
        dict_lib = {}
        suffix = '.zip'
        # Create folders
        for i in range(0, n_files):
            folder_name = self.title + f'_{i}.zip' 
            folder_path = os.path.join(directory, folder_name)
            if os.path.exists(folder_path):
                os.remove(folder_path)
            dict_lib[i] = folder_path   
        return dict_lib    
        
    def run(self, eval_opponent_random):
        self.eval_opponent_random =  eval_opponent_random
        n_gens = self.n_gens
        for gen in range(0, n_gens+1):
            print("gen", gen)
            if gen == 0:
                # intialise training environemnt with a random opponent for the first generation.
                env = texas_holdem.env(self.obs_type, render_mode  = "rgb_array")
                env.OPPONENT.policy = 'random'
                env.AGENT.model = self.base_model
                env.AGENT.policy = self.env.AGENT.policy
                
                # train 
                print("trainin", gen, self.na_key)
                callback_train = CustomLoggerCallback()
                env.AGENT.model.learn(total_timesteps = self.learning_steps, dumb_mode = False, callback=callback_train , progress_bar=True)
                self.metric_dicts.update_train_metrics_from_callback(gen, callback_train)
                # save policy 
                root = self.gen_lib[gen]
                env.AGENT.model.save(path=root, include = ['policy'])
                env.reset()
                 
                # intialise evaluation environemnt with a random opponent for the first generation.
                print("evaluating", gen)
                Eval_env = texas_holdem.env(self.obs_type, render_mode = "rgb_array")
                Eval_env = Monitor(Eval_env)
                Eval_env.OPPONENT.policy = 'random'
                Eval_env.AGENT.policy = self.env.AGENT.policy
                
                # evaulate and update data storage object
                mean_reward_ag,episode_rewards_ag, episode_lengths, rewards_op= evaluate_policy(env.AGENT.model, Eval_env, return_episode_rewards =True, n_eval_episodes = self.n_eval_episodes) 
                percentages_ag,  percentages_op = extract_opt_act(Eval_env)
                self.metric_dicts.update_eval_metrics_from_ep_rewards(gen = gen, mean_reward = mean_reward_ag, episode_rewards = episode_rewards_ag, percentages_ag = percentages_ag, percentages_op= percentages_op)
                
                # train random opponent by setting the dumb mode ie random to True. The models policy has already been saved so this training to does not override the learnt parameters.
                print("training randop", self.na_key)
                callback_train_rand_op = CustomLoggerCallback()              
                env.AGENT.model.learn(total_timesteps = self.learning_steps, dumb_mode = True, callback=callback_train_rand_op , progress_bar=True)
                self.metric_dicts_rand.update_train_metrics_from_callback(gen, callback_train_rand_op)
                
                # Evaluate random opponent
                callback_eval_rand_op = CustomLoggerCallback() 
                print("train randop for eval ", self.na_key)
                env.AGENT.model.learn(total_timesteps = self.n_eval_episodes, dumb_mode = True, callback=callback_eval_rand_op , progress_bar=True,)
                mean_reward_rand = callback_eval_rand_op.final_mean_reward
                episode_rewards_rand = callback_eval_rand_op.rewards
                self.metric_dicts_rand.update_eval_metrics_from_ep_rewards(gen = gen, mean_reward = mean_reward_rand, episode_rewards = episode_rewards_rand, percentages_ag = percentages_ag, percentages_op= percentages_op)

            # for the subsequent generations after the first  
            else:
                # intilaise training environment and retive the file path to the policy of the previous generation.
                env = texas_holdem.env(self.obs_type, render_mode  = "rgb_array")
                prev_gen_path = self.gen_lib[gen-1]
                # init agent and oponent models according to RL model type
                if self.model == 'PPO':
                    if self.default_params:
                        env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [64], 'vf': [64]})
                        env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [64], 'vf': [64]})
                    else:
                        env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [256], 'vf': [256]},learning_rate= 0.005778633008004902, n_steps =  self.n_steps,  batch_size = 32, n_epochs= 70, ent_coef=  0.0025, vf_coef=  0.25, clip_range=0.1, max_grad_norm=0.6, gae_lambda = 0.85, normalize_advantage=False)
                    env.AGENT.policy = 'PPO'
                    env.OPPONENT.policy = 'PPO'
                elif self.model == 'A2C':
                    env.AGENT.model  = A2C('MultiInputPolicy', env, optimizer_class= th.optim.RMSprop, activation_fn = nn.ReLU, learning_rate =  0.04568216636850521, n_steps =10000, ent_coef =0.0025, vf_coef = 0.25, use_rms_prop = False,  net_arch = {'pi': [256], 'vf': [256]}, gae_lambda=0.85, normalize_advantage=False, max_grad_norm=0.5)
                    env.OPPONENT.model  = A2C('MultiInputPolicy', env, optimizer_class= th.optim.RMSprop, activation_fn = nn.ReLU, learning_rate =  0.04568216636850521, n_steps =10000, ent_coef =0.0025, vf_coef = 0.25, use_rms_prop = False,  net_arch = {'pi': [256], 'vf': [256]}, gae_lambda=0.85, normalize_advantage=False, max_grad_norm=0.5)
                    env.AGENT.policy = 'A2C'
                    env.OPPONENT.policy = 'A2C'     
                
                # load prev gen params to opponent and agent 
                env.OPPONENT.model.set_parameters(load_path_or_dict= prev_gen_path)
                env.AGENT.model.set_parameters(load_path_or_dict= prev_gen_path)
                
                # train the agent and update the data storage object
                print("train", gen)
                callback_train = CustomLoggerCallback()
                env.AGENT.model.learn(total_timesteps= self.learning_steps, dumb_mode= False, progress_bar=True, callback=callback_train)
                self.metric_dicts.update_train_metrics_from_callback(gen, callback_train)
                
                # save policy                    
                env.AGENT.model.save(self.gen_lib[gen], include = ['policy'])
                env.reset()
                
                #add models into injector and update metrics dict 
                ci = card_injector(env.AGENT, env.OPPONENT, env)
                ci_results  = ci.return_results()
                self.metric_dicts.update_sims(gen, ci_results)
                
                #create evaluation environment 
                print("eval", gen)
                Eval_env = texas_holdem.env(self.obs_type, render_mode = "rgb_array")
                Eval_env = Monitor(Eval_env)
                Eval_env.AGENT.policy = self.base_model.policy
                if eval_opponent_random:     
                    Eval_env.OPPONENT.policy = 'random'
                else:
                    Eval_env.OPPONENT.policy = self.model
                    Eval_env.OPPONENT.model = env.OPPONENT.model
                #evaluate compared to oppponent   
                mean_reward,episode_rewards, episode_lengths, reward_op= evaluate_policy(env.AGENT.model, Eval_env, callback=None, return_episode_rewards =True, n_eval_episodes = self.n_eval_episodes)
                # extract optimal action percentages
                percentages_ag, percentages_op = extract_opt_act(Eval_env)
                # update data storage from evlaution metrics
                self.metric_dicts.update_eval_metrics_from_ep_rewards(gen, mean_reward,episode_rewards, percentages_ag = percentages_ag, percentages_op= percentages_op)
    
                # random agent train
                print("training randop")
                self.metric_dicts_rand.update_train_metrics_from_callback(gen, callback_train_rand_op)
        
                #random agent evlaution get metrics and update 
                print("training randop for eval")
                self.metric_dicts_rand.update_eval_metrics_from_ep_rewards(gen = gen, mean_reward = mean_reward_rand, episode_rewards = episode_rewards_rand, percentages_ag = percentages_ag, percentages_op= percentages_op)
                            
    def get_results(self, graphs):
        # create graphs from the data storgage object.
        if graphs:
            gm = graph_metrics(n_models = self.n_gens+1, storage = self.metric_dicts, storageB= self.metric_dicts_rand, figsize= (10, 8), t_steps = self.learning_steps, overlay= False, e_steps=self.n_eval_episodes, title = self.title, device='pc' )
            gm.print_all_graphs(True, True, True, False, False, False)

def sp_group():
    """
    This function creates an instance of the self_play class with specified parameters,
    runs the self-play training and evaluation loop, and generates evaluation graphs.

    Args:
    - None

    Returns:
    - None
    """
    sp = self_play(10, 30720, 3072, 'PIG', 6003, 'PPO', na_key=None, default_params=True, info='PIG71')
    sp.run(False)
    sp.get_results(graphs=True)

# sp_group() 

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

    def __init__(self, callback, verbose, batch_size, model_type, override_best, obs_type):
        self.callback = callback
        self.verbose = verbose
        self.batch_size = batch_size,
        self.env =None
        self.model_type = model_type
        self.override_best = override_best
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
            sp = self_play(0, 20480, 1, obs_type = self.obs_type, tag = 44, model = self.model_type, na_key = na_key)
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
            study.optimize(self.optimize_agent, callbacks=None, n_trials=120, n_jobs=1, show_progress_bar=True)
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


def hs_group():
    """
    Perform hyperparameter search experiments for a reinforcement learning model and print results.

    This function initializes a hyperparameter search instance with specified parameters,
    conducts a hyperparameter search experiment, and prints the results including graphs.

    Args:
    - None

    Returns:
    - None
    """
    hypsrch = hyperparam_search(callback=None, verbose=True, batch_size=[32, 64, 128, 256, 512, 1024], model_type='A2C', override_best=True, obs_type='72+')
    hypsrch.init_trained_op()
    print(hypsrch.run(print_graphs=True))

# hs_group()

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
    def __init__(self, callback, verbose, model_type, override_best, obs_type):
        self.callback = callback
        self.verbose = verbose
        self.env =None
        self.model_type = model_type
        self.override_best = override_best
        self.obs_type = obs_type

        
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
        self.n
        self.na_gen_0_dict = {}
        na_key = {'pi': [256], 'vf': [256]}
        sp = self_play(0, 30720, 1, obs_type = self.obs_type, tag = 202, model = self.model_type, na_key = na_key)
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
    
    def run(self, print_graphs):
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
        self.envs = obs_type_envs()
        self.train_envs = self.envs.train_envs
        self.eval_envs = self.envs.eval_envs
        
        self.metric_dicts = metric_dicts()
        self.metric_dicts.add_keys_to_metrics_dict(self.train_envs, self.eval_envs)
        
        self.metric_dicts_rand = metric_dicts()
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
        for key in self.train_envs.keys():
            sp = self_play(0, 20480, 1, obs_type = key, tag = key, model = self.model, na_key = self.na)
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
        for id in self.train_envs.keys():
            env = self.train_envs[id]
            env.OPPONENT.policy = 'PPO'
            env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
            env.AGENT.policy = 'PPO'
            env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
            env.OPPONENT.model.set_parameters(load_path_or_dict= self.trained_opponents[id])
        # Initialize models for evaluation environments    
        for id in self.eval_envs.keys():
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
        for id in self.train_envs.keys():  
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
 
        for id in self.eval_envs.keys():
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
    experiment = obs_experiment(total_timesteps=35000, n_eval_episodes=2048, model='PPO')

    # Train opponents for the experiment
    experiment.train_opponents(gen_to_load=0)

    # Initialize agent and opponent models
    experiment.init_agent_opponent_models()

    # Run training and evaluation for the experiment
    print(experiment.agent_train_eval())

    # Get and optionally plot experiment results
    experiment.get_results(graphs=True)

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

class PPO_vs_allops():
    """
    Class to compare a PPO agent against various opponents and display results.

    This class allows you to compare the performance of a PPO agent against different types of opponents
    (e.g., random, human, A2C, heuristic) in a Texas Hold'em environment. It initializes the opponents,
    evaluates the agents, and displays the results in a bar chart.

    Args:
    - eval_steps (int): Number of evaluation steps/games.

    Attributes:
    - rewards (dict): Dictionary to store mean rewards for different opponents.
    - eval_steps (int): Number of evaluation steps/games.
    - eval_steps_human (int): Number of evaluation steps for human opponent.
    - PPO_path (str): Path to the PPO agent's model parameters file.

    Methods:
    - PPO_vs_random(): Compare PPO agent against a random opponent.
    - PPO_vs_human(): Compare PPO agent against a human opponent.
    - PPO_vs_a2c(): Compare PPO agent against an A2C opponent.
    - PPO_vs_heuristic(): Compare PPO agent against a heuristic opponent.
    - bar_chart(): Generate and display a bar chart of mean rewards.
    - run(): Run the experiments and display results.
    """
    def __init__(self, eval_steps):
        self.rewards = {}
        self.eval_steps = eval_steps 
        self.eval_steps_human = 50 
        self.PPO_path = r'S:\MSC_proj\models\PPO72+10defaultFalse_10'

    def PPO_vs_random(self):
        PPO_vs_random = PPO_vs_OPPONENT( '72+', 'random')
        PPO_vs_random.init_eval_env()
        PPO_vs_random.load_params_from_file(self.PPO_path, None)
        PPO_vs_random.load_metric_dicts_storage()
        PPO_vs_random.evaluate(self.eval_steps)
        PPO_vs_random.get_results(True)
        self.rewards['PPO_vs_random'] = PPO_vs_random.mean_reward
        print(PPO_vs_random.mean_reward)

    def PPO_vs_human(self): 
        PPO_vs_human = human_play('72+', self.eval_steps_human, self.PPO_path)
        PPO_vs_human.play()
        self.rewards['PPO_vs_human'] =  PPO_vs_human.mean_reward

    def PPO_vs_a2c(self):
        PPO_vs_a2c = PPO_vs_OPPONENT('72+', 'A2C')
        PPO_vs_a2c.init_eval_env()
        PPO_vs_a2c.load_params_from_file(self.PPO_path, r'S:\MSC_proj\models\A2C72+10defaultFalse_10')
        PPO_vs_a2c.load_metric_dicts_storage()
        PPO_vs_a2c.evaluate(self.eval_steps)
        PPO_vs_a2c.get_results(True)
        self.rewards['PPO_vs_A2C'] = PPO_vs_a2c.mean_reward

    def PPO_vs_heuristic(self):
        PPO_vs_heuristic = PPO_vs_OPPONENT('72+', 'heuristic')
        PPO_vs_heuristic.init_eval_env()
        PPO_vs_heuristic.load_params_from_file(self.PPO_path, None)
        PPO_vs_heuristic.load_metric_dicts_storage()
        PPO_vs_heuristic.evaluate(self.eval_steps)
        PPO_vs_heuristic.get_results(True)
        self.rewards['PPO_vs_heuristic'] = PPO_vs_heuristic.mean_reward

    def bar_chart(self):
        categories = list(self.rewards.keys())
        counts = list(self.rewards.values())
        colors = ['darkblue','mediumblue', 'blue', 'lightblue']
        plt.bar(categories, counts, color = colors)
        plt.xlabel('Opponents')
        plt.ylabel('PPO mean reward')
        plt.title('PPO mean reward vs opponent for ' + str(self.eval_steps) + ' games')

        # Display the chart
        plt.xticks(rotation=45) 
        plt.tight_layout()  
        plt.show()
        plt.savefig('S:\\MSC_proj\\plots')


    def run(self):
        self.PPO_vs_random()
        # self.PPO_vs_a2c()
        # self.PPO_vs_human()
        # self.PPO_vs_heuristic()
        # self.bar_chart()

allops = PPO_vs_allops(10000)      
allops.run()                                                                                                                                                                                    

def PIG_vs_random():
    PPO_path = r'S:\MSC_proj\models\PPOPIG10defaultTruePIG72_10'
    eval_steps = 10000
    PIG_vs_random = PPO_vs_OPPONENT('PIG', 'random')
    PIG_vs_random.init_eval_env()
    PIG_vs_random.load_params_from_file(PPO_path, None)
    PIG_vs_random.load_metric_dicts_storage()
    PIG_vs_random.evaluate(eval_steps)
    PIG_vs_random.get_results(True)
    print(PIG_vs_random.mean_reward)

PIG_vs_random()

class train_convergence_search():
    """
    Class to perform a search for convergence-related hyperparameters in a PPO training setup.

    This class initializes a search for convergence-related hyperparameters using Optuna, specifically
    for training a PPO agent in a Texas Hold'em environment. It defines a custom Optuna callback
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

    def __init__(self, verbose, obs_type):
        self.verbose = verbose
        self.obs_type = obs_type
        self.model = 'PPO'
        self.trial_results = {}
        
    def init_trained_op(self):
        """
        Initialize the trained opponent models using selfplay for the range of network architectures.
        """
        
        self.na_gen_0_dict = {}
        self.na = [{'pi': [64], 'vf': [64]}]
        for na_key in self.na:
            sp = self_play(0, 20480, 1, obs_type = self.obs_type, tag = 19, model = self.model, na_key = na_key)
            sp.run(False)
            self.na_gen_0_dict[str(na_key)] = sp.gen_lib[0]

    def callback(self,trial, study):
        trail = trial
        study = study
        self.trial_results[study.number] = {
        'params': study.params,
        'value': study.value,
    }
        
    def optimize_cb_params(self, trial):
        return {
        'max_no_improvement_evals': trial.suggest_categorical('max_no_improvement_evals', [3000]),    
        'min_evals': trial.suggest_categorical('min_evals',  [2, 3])
        } 
    
    def optimize_cb(self, trial):
        """
        Perform the convergence search using Optuna.
        """
        cb_params = self.optimize_cb_params(trial)
        
        env = texas_holdem.env(self.obs_type, render_mode = "rgb_array")
        env.OPPONENT.policy = 'PPO'
        env.AGENT.policy = 'PPO'
        env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
        env.OPPONENT.model.set_parameters(self.na_gen_0_dict[str(self.na[0])])
        
        self.env = env

        Eval_env = texas_holdem.env(self.obs_type, render_mode = "rgb_array")
        Eval_env = Monitor(Eval_env)
        Eval_env.OPPONENT.policy = 'random'
        Eval_env.AGENT.policy = 'PPO'

        cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals= cb_params['max_no_improvement_evals'], min_evals=cb_params['min_evals'], verbose =1)
        cb = EvalCallback(Eval_env, eval_freq=10000, callback_after_eval=cb, verbose=0, n_eval_episodes= cb_params['max_no_improvement_evals']) 
           
        model = PPO('MultiInputPolicy', self.env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
        model.learn(total_timesteps=40000, dumb_mode = False, progress_bar=self.verbose, callback= cb)
    
        return cb.callback.parent.best_mean_reward
            
    def run(self, print_graphs):
        if __name__ == '__main__':
            study = optuna.create_study(direction= 'maximize')
        try:
            study.optimize(self.optimize_cb, n_trials=8, n_jobs=1, show_progress_bar= True, callbacks = [self.callback])
        except KeyboardInterrupt:
            print('Interrupted by keyboard.')
        print(study.best_params)
        if print_graphs == True:
            optuna.visualization.plot_param_importances(study).show()
            optuna.visualization.plot_optimization_history(study).show() 
    
    def plot_dictionary(self, dict ):
        dictionary = dict
        mean_dict = {}
    
        # Calculate the mean values for duplicate keys
        for key, value in dictionary.items():
            if key in mean_dict:
                mean_dict[key].append(value)
            else:
                mean_dict[key] = [value]
        
        for key in mean_dict:
            mean_dict[key] = sum(mean_dict[key]) / len(mean_dict[key])
        
        # Extract keys and values for plotting
        x_values = list(mean_dict.keys())
        y_values = list(mean_dict.values())
        
        # Create a bar graph
        plt.bar(x_values, y_values)
        plt.xlabel('No improvement steps')
        plt.ylabel('Mean Values')
        plt.title('Number of steps vs mean reward')

        plt.show()
                          
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
        gm.print_select_graphs(False, False, False,False, False, True, False)    
        return self.results
# kl = kl_div_test('72+',10)
# print(kl.run()) 
class NE_tool():
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
        self.n_eval_episodes = 10000
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
        plt.tight_layout()  # Optional: improves plot layout
        plt.show()

# NET = NE_tool('72+', 10)
# NET.run()
# NET.print_graph()