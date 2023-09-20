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

        # initialize the agent model depending on if default hyperparameters are required. set oppoents policy as random for the first generation training.
        self.env = texas_holdem.env(self.obs_type, render_mode  = "rgb_array")
        self.env.OPPONENT.policy = 'random'
        if self.model == 'PPO':
            if default_params:
                 self.base_model = PPO('MultiInputPolicy', self.env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [64], 'vf': [64]})
            else:      
                 self.base_model = PPO('MultiInputPolicy', self.env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh, net_arch = {'pi': [256], 'vf': [256]},learning_rate= 0.005778633008004902, n_steps =  self.n_steps,  batch_size = 32, n_epochs= 70, ent_coef=  0.0025, vf_coef=  0.25, clip_range=0.1, max_grad_norm=0.6, gae_lambda = 0.85, normalize_advantage=False)
            self.env.AGENT.policy = 'PPO'
        elif self.model == 'A2C':
            self.base_model = A2C('MultiInputPolicy',self.env, optimizer_class= th.optim.RMSprop, activation_fn = nn.ReLU, learning_rate =  0.04568216636850521, n_steps =10000, ent_coef =0.0025, vf_coef = 0.25, use_rms_prop = False,  net_arch = {'pi': [256], 'vf': [256]}, gae_lambda=0.85, normalize_advantage=False, max_grad_norm=0.5)
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
    sp = self_play(5, 30720*2, 3072, 'PIG', 60072, 'PPO', na_key=None, default_params=True, info='PIG73')
    sp.run(False)
    sp.get_results(graphs=True)

sp_group() 