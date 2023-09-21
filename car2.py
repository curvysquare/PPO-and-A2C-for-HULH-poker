import numpy as np   
import texas_holdem_mod as texas_holdem
import rlcard 
from rlcard.utils.utils import print_card as prnt_cd
import texas_holdem_mod as texas_holdem
from rlcard.utils.utils import print_card as prnt_cd
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os 
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from gymnasium import Env
import optuna
import gym
import numpy as np
import torch as th
from torch import nn
from tabulate import tabulate
import pandas as pd
# from rlcard.agents.human_agents.nolimit_holdem_human_agent import HumanAgent
from classmaker import graph_metrics
from classmaker import obs_type_envs
from classmaker import metric_dicts

from injector import card_injector

def extract_opt_act(env):

    acts = env.opt_acts 
    
    total_elements = len(acts)
    ones_count = acts.count(1)  # Count the occurrences of 1
    
    if total_elements == 0:
        return 0.0
    
    percentage = (ones_count / total_elements) * 100
    
    percentages = []
    for i, act in enumerate(acts):
        ones_count = acts[:i+1].count(1)
        percentage = (ones_count / (i + 1)) * 100
        percentages.append(percentage)
    return percentages, percentages[-1]
    
    

def list_statistics(data):
    mean = np.mean(data)
    max_val = np.max(data)
    min_val = np.min(data)
    data_range = np.ptp(data)  # Peak-to-peak (range) value

    return mean, max_val, min_val, data_range

def dict_to_list(dictionary):
    activation_fn = dictionary.pop('activation_fn', None)
    net_arch = dictionary.pop('net_arch', None)
    result_list = []

    if activation_fn is not None:
        result_list.append(f"activation_fn = {activation_fn}")

    if net_arch is not None:
        result_list.append(f"net_arch = {net_arch}")

    for key, value in dictionary.items():
        result_list.append(f"{key} = {value}")

    return result_list

def convert_np_float32_to_float(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, np.float32):
            dictionary[key] = float(value)
    return dictionary

class CustomLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomLoggerCallback, self).__init__(verbose)
        self.losses = []
        self.moving_mean_reward = []
        self.rewards = [] 
        self.moving_total = [0]
        self.step_list = []
        self.opt_acts_over_eps =[]
        
        # self.op_moving_mean_reward= []
        # self.op_rewards = []
        # self.op_moving_total = [0]
        if len( self.moving_mean_reward)>1:
            self.final_mean_reward = self.moving_mean_reward[-1]
        

        
    def _on_step(self) -> bool:
        if hasattr(self.model, 'loss'):
            if self.model.loss != None:
                loss = self.model.loss.item()
                loss = float(loss)
                self.losses.append(loss)     
        self.step_list.append(self.num_timesteps)
        self.rewards.append(self.model.env.buf_rews[0])
        self.moving_mean_reward.append(np.mean(self.rewards))
        self.moving_total.append(self.moving_total[-1] + self.model.env.buf_rews[0])
        if len( self.moving_mean_reward)>1:
            self.final_mean_reward = self.moving_mean_reward[-1]
        

        
        
        # self.op_rewards.append(-1*self.model.env.buf_rews[0])
        # self.op_moving_mean_reward.append(np.mean(self.op_rewards))
        # self.op_moving_total.append(self.op_moving_total[-1] + self.model.env.buf_rews[0]*-1)

        # print("step", self.num_timesteps, "call", self.n_calls, "rewards", self.model.env.buf_rews[0])
        # reward = self.model.env.buf_rews

            
        return True
                      
class BatchMultiplier:
    def __init__(self, batch_sizes):
        self.batch_sizes = batch_sizes[0]

    def gcd(self, a, b):
        # Calculate the greatest common divisor using Euclidean algorithm
        while b != 0:
            a, b = b, a % b
        return a

    def lcm(self, a, b):
        # Calculate the least common multiple using the formula: LCM(a, b) = (a * b) / GCD(a, b)
        return (a * b) // self.gcd(a, b)

    def lcm_of_list(self, numbers):
        # Calculate the least common multiple of a list of numbers
        result = 1
        for num in numbers:
            result = self.lcm(result, num)
        return result

    def generate_divisible_integers(self):
        # Find the LCM of the batch sizes
        lcm_batch_sizes = self.lcm_of_list(self.batch_sizes)

        # Generate a list of integers that are multiples of the LCM
        divisible_integers = [i * lcm_batch_sizes for i in range(1, 6)]

        return divisible_integers
    
 
# results for 72§ = {'max_no_improvement_evals': 45, 'min_evals': 35}

class self_play():
    def __init__(self, n_gens, learning_steps, n_eval_episodes, obs_type, tag, model, na_key): 
        self.n_gens = n_gens
        self.learning_steps = learning_steps
        self.n_eval_episodes = n_eval_episodes
        self.obs_type  = obs_type
        env = texas_holdem.env(self.obs_type, render_mode  = "rgb_array")
        env.OPPONENT.policy = 'random'
        self.env = env
        self.tag = tag
        self.model = model
        self.na_key = na_key
        if self.model == 'PPO':
            # self.base_model = PPO('MultiInputPolicy', self.env, optimizer_class = th.optim.Adam,activation_fn= nn.ReLU,net_arch=  self.na_key,learning_rate=  0.07100101556878591, n_steps = 2048, batch_size = 128, n_epochs=  31, ent_coef=  0.000125, vf_coef=  0.25)
            self.base_model = PPO('MultiInputPolicy', self.env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na_key, n_steps=20480)
            self.env.AGENT.policy = 'PPO'
        elif self.model == 'A2C':
            self.base_model = A2C('MultiInputPolicy', self.env, optimizer_class= th.optim.RMSprop, activation_fn= nn.Tanh, net_arch=  self.na_key, learning_rate=0.0008042927331413859, n_steps=2048, ent_coef = 0.0025, vf_coef =0.8, use_rms_prop=True)
            self.env.AGENT.policy = 'A2C'
        
        self.gen_lib = self.create_files(self.n_gens + 1) 
        
        self.gen_keys = []
        for gen in range(n_gens+1):
            self.gen_keys.append(gen)
            
        
        self.metric_dicts = metric_dicts(self.n_gens)
        self.metric_dicts.add_keys_to_metrics(self.gen_keys)
        
        self.metric_dicts_rand = metric_dicts(self.n_gens)
        self.metric_dicts_rand.add_keys_to_metrics(self.gen_keys)
        

  
    def create_files(self, n_files):
        directory = '/Users/rhyscooper/Desktop/MSc Project/Pages/models'
        dict_lib = {}
        suffix = '.zip'
        # Create ten folders
        for i in range(0, n_files):
            folder_name = str(self.tag) + f'_{i}.zip' 
            folder_path = os.path.join(directory, folder_name)
            if os.path.exists(folder_path):
                os.remove(folder_path)
            # folder_path = os.path.join(folder_path, suffix)
            dict_lib[i] = folder_path
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)
            # else:
            #     pass    
                
        return dict_lib    
        
    def run(self, eval_opponent_random):
        self.eval_opponent_random =  eval_opponent_random
        n_gens = self.n_gens
        for gen in range(0, n_gens+1):
            print("gen", gen)
            if gen == 0:
                env = self.env
                env.AGENT.model = self.base_model
                env.AGENT.policy = self.env.AGENT.policy
                
                
                print("trainin", gen, self.na_key)
                callback_train = CustomLoggerCallback()
                env.AGENT.model.learn(total_timesteps = self.learning_steps, dumb_mode = False, callback=callback_train , progress_bar=True)
                self.metric_dicts.update_train_metrics_from_callback(gen, callback_train)
                
                print("training randop", self.na_key)
                callback_train_rand_op = CustomLoggerCallback()              
                env.AGENT.model.learn(total_timesteps = self.learning_steps, dumb_mode = True, callback=callback_train_rand_op , progress_bar=True)
                self.metric_dicts_rand.update_train_metrics_from_callback(gen, callback_train_rand_op)
 
                root = self.gen_lib[gen]
                env.AGENT.model.save(path=root, include = ['policy'])
                env.reset()
                 
                
                Eval_env = texas_holdem.env(self.obs_type, render_mode = "rgb_array")
                Eval_env = Monitor(Eval_env)
                Eval_env.OPPONENT.policy = 'random'
                Eval_env.AGENT.policy = self.env.AGENT.policy
                

                mean_reward,episode_rewards, episode_lengths= evaluate_policy(env.AGENT.model, Eval_env, return_episode_rewards =True, n_eval_episodes = self.n_eval_episodes) 
                # percentages, p = extract_opt_act(Eval_env)
                self.metric_dicts.update_eval_metrics_from_ep_rewards(gen = gen, mean_reward = mean_reward, episode_rewards = episode_rewards, percentages = None)
                
                
                
                callback_eval_rand_op = CustomLoggerCallback() 
                print("train randop for eval ", self.na_key)
                env.AGENT.model.learn(total_timesteps = len(episode_rewards), dumb_mode = True, callback=callback_eval_rand_op , progress_bar=True,)
                
                mean_reward = callback_eval_rand_op.final_mean_reward
                episode_rewards = callback_eval_rand_op.rewards
                self.metric_dicts_rand.update_eval_metrics_from_ep_rewards(gen = gen, mean_reward = mean_reward, episode_rewards = episode_rewards, percentages=None)

                
                
                
            else:
                env = self.env
                env.reset()
                prev_gen_path = self.gen_lib[gen-1]
                env.OPPONENT.policy = self.model
                env.OPPONENT.model = self.base_model
                env.OPPONENT.model.set_parameters(load_path_or_dict= prev_gen_path)
                env.AGENT.model = self.base_model
                
                callback_train = CustomLoggerCallback()
                env.AGENT.model.learn(total_timesteps= self.learning_steps, dumb_mode= False, progress_bar=True, callback=callback_train)
                self.metric_dicts.update_train_metrics_from_callback(gen, callback_train)
                
                callback_train_rand_op = CustomLoggerCallback()              
                env.AGENT.model.learn(total_timesteps = self.learning_steps, dumb_mode = True, callback=callback_train_rand_op , progress_bar=True,)
                self.metric_dicts_rand.update_train_metrics_from_callback(gen, callback_train)
        
        
        
                env.AGENT.model.save(self.gen_lib[gen], include = ['policy'])
                
                
                
                
                env.reset()
                
                #add models into injector and update metrics dict 
                ci = card_injector(env.AGENT, env.OPPONENT, env)
                ci_results  = ci.return_results()
                self.metric_dicts.update_sims(gen, ci_results)
                
                
                #evaluate compared to oppponent
                
                Eval_env = texas_holdem.env(self.obs_type, render_mode = "rgb_array")
                Eval_env = Monitor(Eval_env)
                Eval_env.AGENT.policy = self.base_model.policy
                if eval_opponent_random:     
                    Eval_env.OPPONENT.policy = 'random'
                else:
                    Eval_env.OPPONENT.policy = self.model
                    Eval_env.OPPONENT.model = self.base_model
                    
        
                mean_reward,episode_rewards, episode_lengths= evaluate_policy(env.AGENT.model, Eval_env, callback=None, return_episode_rewards =True, n_eval_episodes = self.n_eval_episodes)
                # percentages, p = extract_opt_act(Eval_env)
                self.metric_dicts.update_eval_metrics_from_ep_rewards(gen, mean_reward,episode_rewards, percentages = None)
                
                #random agent evlaution get metrics and update 
                callback_eval_rand_op = CustomLoggerCallback() 
                env.AGENT.model.learn(total_timesteps = len(episode_rewards), dumb_mode = True, callback=callback_eval_rand_op , progress_bar=True,)
                
                mean_reward = callback_eval_rand_op.final_mean_reward
                episode_rewards = callback_eval_rand_op.rewards
                self.metric_dicts_rand.update_eval_metrics_from_ep_rewards(gen = gen, mean_reward = mean_reward, episode_rewards = episode_rewards, percentages = None)
                            
    def get_results(self, graphs):
        print(self.metric_dicts.gen_train_final_mean_reward)
        print(self.metric_dicts.gen_eval_final_mean_reward)
        
        if graphs:
            gm = graph_metrics(n_models = self.n_gens+1, storage = self.metric_dicts, storageB= self.metric_dicts_rand, figsize= (10, 8), t_steps = self.learning_steps )
            gm.print_all_graphs(True, True)

class hyperparam_search(BatchMultiplier):
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
        self.na_gen_0_dict = {}
        na = self.net_arch
        for na_key in na:
            sp = self_play(0, 20480, 1, obs_type = self.obs_type, tag = 19, model = self.model_type, na_key = na_key)
            sp.run(False)
            self.na_gen_0_dict[str(na_key)] = sp.gen_lib[0]
            
        # self.root = sp.gen_lib[0]
        
    def optimize_ppo(self, trial):
        # self.net_arch = [{'pi':[64,64], 'vf': [64,64]}, {'pi':[256], 'vf': [256]}, {'pi':[256,128], 'vf': [256,128]}]
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
        
        if params['use_rms_prop'] == False:
                    params['optimizer_class']: trial.suggest_categorical('optimizer_class', [th.optim.Adam, th.optim.SGD])
                    
           
        
        return params
    
    def custom_exception_handler(self,func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the exception or do other handling if needed
                print(f"Exception caught: {e}")
                return None

        return wrapper

    def optimize_agent(self, trial):
        callback = self.callback
        
    # inti params depending on model
        if self.model_type == 'PPO':
            model_params = self.optimize_ppo(trial)
        
        if self.model_type == 'A2C':    
            model_params = self.optimize_A2C(trial) 
        
    # init env    
        env = texas_holdem.env(self.obs_type, render_mode="rgb_array")
        

    # inti eval enc
        Eval_env = texas_holdem.env(self.obs_type, render_mode="rgb_array")
        

        # Eval_env.OPPONENT.policy = 'random'
        
    # init policies     
        if self.model_type == 'PPO':
            env.AGENT.policy = 'PPO'
            env.OPPONENT.policy = 'PPO'
            Eval_env.AGENT.policy = 'PPO'
            Eval_env.OPPONENT.policy = 'PPO'
            
        elif self.model_type == 'A2C':
            env.AGENT.policy = 'A2C'
            env.OPPONENT.policy = 'A2C'
            Eval_env.AGENT.policy = 'A2C'
            Eval_env.OPPONENT.policy = 'A2C'
    
              
        if self.model_type == 'PPO':
            env.AGENT.model = PPO('MultiInputPolicy', env, verbose=0, **model_params)
            env.OPPONENT.model = PPO('MultiInputPolicy', env, verbose=0, **model_params)
            env.OPPONENT.model.set_parameters(self.na_gen_0_dict[str(model_params['net_arch'])])
        
            # Eval_env.OPPONENT.model =  PPO('MultiInputPolicy', env, verbose=0, **model_params)
            # Eval_env.OPPONENT.model.set_parameters(self.na_gen_0_dict[str(model_params['net_arch'])])
            
            Eval_env.OPPONENT.policy = 'random'
            
            
            
        if self.model_type == 'A2C':
            env.AGENT.model = A2C('MultiInputPolicy', self.env, verbose=0, **model_params) 
            env.OPPONENT.model = A2C('MultiInputPolicy', self.env, verbose=0, **model_params)
            env.OPPONENT.model.set_parameters(self.na_gen_0_dict[str(model_params['net_arch'])])
            
            Eval_env.OPPONENT.model =  A2C('MultiInputPolicy', env, verbose=0, **model_params)
            Eval_env.OPPONENT.model.set_parameters(self.na_gen_0_dict[str(model_params['net_arch'])])
            
               
        Eval_env = Monitor(Eval_env)
        self.env = env
        cb = StopTrainingOnNoModelImprovement(1000, 75)
            
        cb = EvalCallback(Eval_env, eval_freq=10000, callback_after_eval= cb, verbose=0, n_eval_episodes = 1000)
        try: 
            env.AGENT.model.learn(total_timesteps=40000, callback= cb, progress_bar=False, dumb_mode=False)
            mean_reward, _ = evaluate_policy(self.env.AGENT.model, Eval_env, n_eval_episodes=1000)
        except ValueError:
            mean_reward = 0.0
        return mean_reward
    
    def run(self, print_graphs):
        
        self.optimize_agent = self.custom_exception_handler(self.optimize_agent)
        if __name__ == '__main__':
            study = optuna.create_study(direction= 'maximize')
        try:
            study.optimize(self.optimize_agent, callbacks=None, n_trials=120, n_jobs=1, show_progress_bar=True)
        except KeyboardInterrupt:
            print('Interrupted by keyboard.')
 
        self.best_trial = study.best_trial
        self.best_params = study.best_params
        
        # if self.model_type == 'PPO':
        #     self.best_model = PPO('MultiInputPolicy', self.env, verbose=0, **self.best_params)
            
        # elif self.model_type == 'A2C':
        #     self.best_model = A2C('MultiInputPolicy', self.env, verbose=0, **self.best_params)    
        
        if print_graphs == True:
            optuna.visualization.plot_param_importances(study).show()
            optuna.visualization.plot_optimization_history(study).show()
            

        return self.best_trial, self.best_params
    def best_model_train_and_test(self,):
        env = texas_holdem.env()
        self.env = env
        
        Eval_env = texas_holdem.env()
        Eval_env = Monitor(Eval_env)

        env.OPPONENT.policy = 'random'
        Eval_env.OPPONENT.policy = 'random'
        
        if self.model_type == 'PPO':
            env.AGENT.policy = 'PPO'
            Eval_env.AGENT.policy = 'PPO'
            
        elif self.model_type == 'A2C':
            env.AGENT.policy = 'A2C'
            Eval_env.AGENT.policy = 'A2C'
        
        if self.override_best:
            self.best_model = PPO('MultiInputPolicy', env, verbose=0, batch_size = 64, n_steps =2048, learning_rate = 0.052224490223871795, n_epochs = 11, ent_coef = 0.000125, vf_coef = 0.25, activation_fn = th.nn.modules.activation.ReLU, optimizer_class = th.optim.Adam, net_arch =  {'pi': [256], 'vf': [256]} )
                              
        # batch_size = 64, n_steps =2048, learning_rate = 0.052224490223871795, n_epochs = 11.0, ent_coef = 0.000125, vf_coef = 0.25, activation_fn = th.nn.modules.activation.ReLU, optimizer_class = th.optim.Adam, net_arch =  {'pi': [256], 'vf': [256]})
        # callback1 = StopTrainingOnNoModelImprovement(max_no_improvement_evals=45, min_evals=35)
        
        callback2 = CustomLoggerCallback()
        callback1 = EvalCallback(Eval_env, eval_freq=30, callback_after_eval=callback2, verbose=0, n_eval_episodes = 10)
        self.best_model.learn(total_timesteps=2048, callback= callback1, progress_bar=True)
        training_losses = callback2.losses
        print("Training loss", training_losses)
        training_rewards = callback2.mean_rewards
        print("training rewads", len(training_rewards), 2048/64, 2048/30)

        # plt.plot(range(len(training_losses)), training_losses)
        # plt.xlabel("Training Steps")
        # plt.ylabel("loss")
        # plt.title("loss vs. episodes")
        # plt.grid(True)
        # plt.show()        



        plt.plot(range(len(training_rewards)), training_rewards)
        plt.xlabel("Training Steps")
        plt.ylabel("rewards")
        plt.title("rewards vs steps")
        plt.grid(True)
        plt.show()        

        
        
        # Eval_env = texas_holdem.env()
        # Eval_env = Monitor(Eval_env)
        # Eval_env.OPPONENT.policy = 'random'
        # Eval_env.AGENT.policy = 'PPO'    
             
        # mean_reward, _ = evaluate_policy(self.best_model, Eval_env, n_eval_episodes=30)    

# self_play_object_124 = self_play(4, 100, 100, '72+', 12, 'PPO', na_key ={'pi': [64], 'vf': [64]} )
# self_play_object_124.run(False)
# self_play_object_124.get_results(graphs = True)

# callback1 = StopTrainingOnNoModelImprovement(max_no_improvement_evals=45, min_evals=35)         
# hypsrch = hyperparam_search(callback= None, verbose=True, batch_size=[32, 64, 128, 256, 512, 1024], model_type= 'PPO', override_best=True, obs_type= '72+')
# hypsrch.init_trained_op()
# print(hypsrch.run(print_graphs= True))

# Trial 110 finished with value: 3.0325 and parameters: {'batch_size': 512, 'n_steps': 2048, 'learning_rate': 0.05371465474343706, 'n_epochs': 50.0, 'ent_coef': 0.000125, 'vf_coef': 0.25, 'activation_fn': <class 'torch.nn.modules.activation.ReLU'>, 'optimizer_class': <class 'torch.optim.adam.Adam'>, 'net_arch': {'pi': [256, 128], 'vf': [256, 128]}}. Best is trial 110 with value: 3.0325.
# 120 trials took 3.5 hours.

#  Trial 43 finished with value: 5.679 and parameters: {'batch_size': 32, 'n_steps': 3072, 'learning_rate': 0.034440481970522255, 'n_epochs': 90.0, 'ent_coef': 0.0025, 'vf_coef': 0.25, 'activation_fn': <class 'torch.nn.modules.activation.Tanh'>, 'optimizer_class': <class 'torch.optim.adam.Adam'>, 'net_arch': {'pi': [256], 'vf': [256]}}. Best is trial 43 with value: 5.679.

# Trial 40 finished with value: 4.02 and parameters: {'batch_size': 32, 'n_steps': 3072, 'learning_rate': 0.005778633008004902, 'n_epochs': 70.0, 'ent_coef': 0.0025, 'vf_coef': 0.25, 'activation_fn': <class 'torch.nn.modules.activation.Tanh'>, 'optimizer_class': <class 'torch.optim.adam.Adam'>, 'net_arch': {'pi': [256], 'vf': [256]}}. Best is trial 40 with value: 4.02.

# Trial 87 finished with value: 3.159 and parameters: {'batch_size': 256, 'n_steps': 3072, 'learning_rate': 0.01017124538443069, 'n_epochs': 80.0, 'ent_coef': 0.02, 'vf_coef': 0.85, 'activation_fn': <class 'torch.nn.modules.activation.Tanh'>, 'optimizer_class': <class 'torch.optim.sgd.SGD'>, 'net_arch': {'pi': [256, 128], 'vf': [256, 128]}}. Best is trial 87 with value: 3.159.

# Trial 68 finished with value: 3.164 and parameters: {'batch_size': 128, 'n_steps': 1024, 'learning_rate': 0.04345527858082922, 'n_epochs': 100.0, 'ent_coef': 0.02, 'vf_coef': 0.8, 'activation_fn': <class 'torch.nn.modules.activation.ReLU'>, 'optimizer_class': <class 'torch.optim.adam.Adam'>, 'net_arch': {'pi': [256, 128], 'vf': [256, 128]}}
# best_hyps = [batch_size = 64, n_steps =2048, learning_rate = 0.052224490223871795, n_epochs = 11.0, ent_coef = 0.000125, vf_coef = 0.25, activation_fn = torch.nn.modules.activation.ReLU, optimizer_class = torch.optim.adam.Adam, net_arch =  {'pi': [256], 'vf': [256]}]
# hypsrch.best_model = PPO('MultiInputPolicy', hypsrch.env, verbose=0, batch_size = 64, n_steps =2048, learning_rate = 0.052224490223871795, n_epochs = 11.0, ent_coef = 0.000125, vf_coef = 0.25, activation_fn = th.nn.modules.activation.ReLU, optimizer_class = th.optim.Adam, net_arch =  {'pi': [256], 'vf': [256]})


# Trial 21 finished with value: 3.41 and parameters: {'batch_size': 128, 'n_steps': 2048, 'learning_rate': 0.07100101556878591, 'n_epochs': 31.0, 'ent_coef': 0.000125, 'vf_coef': 0.25, 'activation_fn': <class 'torch.nn.modules.activation.ReLU'>, 'optimizer_class': <class 'torch.optim.adam.Adam'>, 'net_arch': {'pi': [256], 'vf': [256]}}. Best is trial 21 with value: 3.41.  

# Trial 81 finished with value: 3.177 and parameters: {'batch_size': 512, 'n_steps': 1024, 'learning_rate': 0.026126326488343357, 'n_epochs': 80.0, 'ent_coef': 0.02, 'vf_coef': 0.8, 'activation_fn': <class 'torch.nn.modules.activation.ReLU'>, 'optimizer_class': <class 'torch.optim.adam.Adam'>, 'net_arch': {'pi': [64, 64], 'vf': [64, 64]}} 

#A2C hypsearch 


# A2C_search = hyperparam_search(callback= callback1, verbose=True, batch_size = None, model_type = 'A2C', override_best= False)
# print(A2C_search.run(print_graphs= True))

# [I 2023-08-08 19:15:33,100] Trial 187 finished with value: 3.511 and parameters: {'activation_fn': <class 'torch.nn.modules.activation.Tanh'>, 'net_arch': {'pi': [64, 64], 'vf': [64, 64]}, 'learning_rate': 0.0008042927331413859, 'n_steps': 1000, 'ent_coef': 0.0025, 'vf_coef': 0.8, 'use_rms_prop': False, 'optimizer_class': <class 'torch.optim.rmsprop.RMSprop'>}. Best is trial 187 with value: 3.511.
# 4.5 hours, 50,000 trials, 1000 eval eps.




class epoch_search():
    def __init__(self, callback):
        self.callback = callback
  
    def params_test(self, trial):
        return {
        'n_epochs': int(trial.suggest_categorical('n_epochs', [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100])),
        }
            
    def optimize_PPO(self, trial):
        callback = self.callback
            
        env = texas_holdem.env()
        env.OPPONENT.policy = 'random'
        env.AGENT.policy = 'PPO'
        
        self.env = env
        
        Eval_env = texas_holdem.env()
        Eval_env = Monitor(Eval_env)
        Eval_env.OPPONENT.policy = 'random'
        Eval_env.AGENT.policy = 'PPO'
        
        model_params =  self.params_test(trial)
        
                
        model = PPO('MultiInputPolicy', self.env, verbose=0,batch_size = 64, n_steps =2048, learning_rate = 0.052224490223871795,
                    ent_coef = 0.000125, vf_coef = 0.25, activation_fn = th.nn.modules.activation.ReLU, optimizer_class = th.optim.Adam, 
                    net_arch =  {'pi': [256], 'vf': [256]}, **model_params)
        
        cb = EvalCallback(Eval_env, eval_freq=10000, callback_after_eval=self.callback, verbose=0, n_eval_episodes = 100) 
        model.learn(total_timesteps=20480, callback= None, progress_bar=False)
        mean_reward, _ = evaluate_policy(model, Eval_env, n_eval_episodes=1000)

        return mean_reward
    
    def run(self, print_graphs):
        

        if __name__ == '__main__':
            study = optuna.create_study(direction= 'maximize')
        try:
            study.optimize(self.optimize_PPO, callbacks=None, n_trials=5, n_jobs=1, show_progress_bar=True)
        except KeyboardInterrupt:
            print('Interrupted by keyboard.')
 
        
        self.best_trial = study.best_trial
        self.best_params = study.best_params
        self.best_model = PPO('MultiInputPolicy', self.env, verbose=0, **self.best_params)
        
        if print_graphs == True:
            optuna.visualization.plot_param_importances(study).show()
            optuna.visualization.plot_optimization_history(study).show()
            

        return self.best_trial, self.best_params,     
    
# epoch_search_optuna = epoch_search(callback=None)
# print(epoch_search_optuna.run(print_graphs=True))

class main_interface():
    def __init__(self, verbose):
        self.verbose = verbose 
    
class file_creater():
    def __init__(self, n_files) -> None:
        self.n_files = n_files
    def make_files(self):    
        directory = '/Users/rhyscooper/Desktop/MSc Project/Pages/models'
        dict_lib = {}
        suffix = '.zip'
        # Create ten folders
        for i in range(0, self.n_files):
            folder_name = f'{i}.zip'
            folder_path = os.path.join(directory, folder_name)
            # folder_path = os.path.join(folder_path, suffix)
            dict_lib[i] = folder_path
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)
            # else:
            #     pass    
                
        return dict_lib     

    def chip_reader(self,obs):
        obs = obs[0]
        obs = obs['observation']
        chips = obs[52:]
        chips_rounds = {}
        chips_rounds['r1'] = chips[:5]
        chips_rounds['r2'] = chips[5:10]
        chips_rounds['r3'] = chips[10:15]
        chips_rounds['r4'] = chips[15:20]
        
        return chips_rounds

    def env_readout_loop(self):
        env = texas_holdem.env()
        env.OPPONENT.policy = 'random'
        env.AGENT.policy = 'DQN'
        env.AGENT.model = DQN('MultiInputPolicy', env, verbose=0)
        
        obs = env.reset()
        print("1", chip_reader(obs))
        action = 1
        obs = env.step(action)
        print("2", chip_reader(obs))
        action = env.AGENT.get_action(obs)
        obs = env.step(action)
        print("3", chip_reader(obs))
        action = env.AGENT.get_action(obs)
        obs = env.step(action)
        print("4", chip_reader(obs))
        action = env.AGENT.get_action(obs)
        print("5", chip_reader(obs))
        obs = env.step(action)
        action = env.AGENT.get_action(obs)

    def env_readout(self,times):
            for i in range(0, times):
                env = texas_holdem.env()
                env.OPPONENT.policy = 'random'
                env.AGENT.policy = 'PPO'
                env.AGENT.model = PPO('MultiInputPolicy', env, verbose=0)
                
                obs = env.reset()
                print("first obs from reset", obs)
                # print(chip_reader(obs))
                # print("pointer:", env.game_pointer_pu, "agent selected", env.agent_selection )
                
                # print(env.AGENT.get_action_mask())
                action = env.AGENT.get_action(obs)
                print("actions", action)
                
                obs = env.step(action)
        
            if obs[2] == False:
                action = env.AGENT.get_action(obs)
                obs = env.step(action)
                
                # print("2dn", chip_reader(obs))
                print("step", obs)
                # print("pointer:", env.game_pointer_pu, "agent selected", env.agent_selection )
                
                # print(env.step(1))
                # print("pointer:", env.game_pointer_pu, "agent selected", env.agent_selection )
                
                # print(env.step(1))
                # print("pointer:", env.game_pointer_pu, "agent selected", env.agent_selection )

    def learn_and_plot(self, n_steps, eval_freq, n_eval_episodes, total_timesteps, batch_size):

        
        env = texas_holdem.env()
        env.OPPONENT.policy = 'random'
        env.AGENT.policy = 'PPO'
    

        Eval_Env = texas_holdem.env()
        Eval_Env = Monitor(Eval_Env)
        evalcb = EvalCallback(Eval_Env, n_eval_episodes = n_eval_episodes, eval_freq= eval_freq, log_path='/Users/rhyscooper/Desktop/MScProject/Pages/logs', verbose =0)
        Eval_Env.OPPONENT.policy = 'random'
        Eval_Env.AGENT.policy = 'PPO' 

        # env =  make_vec_env(env, n_envs=1)
            
        # env.AGENT.model = PPO('MultiInputPolicy', env, verbose=1, n_steps = n_steps, batch_size= batch_size, n_epochs =1)
        # env.AGENT.model.learn(total_timesteps=total_timesteps, progress_bar=True, callback= evalcb)
        
        model = PPO('MultiInputPolicy', env, verbose=1, n_steps = n_steps, batch_size= batch_size, n_epochs =1)
        model.learn(total_timesteps=total_timesteps, progress_bar=True, callback= evalcb)
    

        results = np.load('/Users/rhyscooper/Desktop/MScProject/Pages/logs/evaluations.npz', 'r+')
        rewards, steps, eplen =  results['results'], results['timesteps'], results['ep_lengths']
        rewards= [item for sublist in rewards for item in sublist]
        # print(len(rewards), "rewards", rewards)
        env.plot_cumulative_reward(rewards)
        return rewards
   

            
        
                
#model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam,activation_fn= nn.ReLU,net_arch=  {'pi': [256], 'vf': [256]},learning_rate=  0.07100101556878591, n_steps = 2048, batch_size = 128, n_epochs=  31, ent_coef=  0.000125, vf_coef=  0.25)

# big kuhuna: 160 min eta
# self_play_object = self_play(3, 200000, 100000)

# self_play_object_124 = self_play(1, 4000, 1000, '124', 6, 'PPO')
# self_play_object_124.run(False)
# self_play_object_124.get_results(graphs = True)
# print(self_play_object.get_results(graphs = True)) 

# 2 hours for 390, 1000      
# 0: 1.0374258, 1: -0.01631258, 2: 0.0022421079}
#{0: 7.25, 1: 3.0737, 2: 3.049125}                
     
def simp_DQN():
    env = texas_holdem.env()
    env.OPPONENT.policy = 'random'
    env.AGENT.policy = 'DQN'

    env.AGENT.model = DQN('MultiInputPolicy', env, verbose=0)
    env.AGENT.model.learn(total_timesteps=3, progress_bar=True )
    # env.AGENT.model.learn(total_timesteps=20, progress_bar=True)
    
# simp_DQN()





def simp_PPO(learning_steps, model):
    env = texas_holdem.env('72+', render_mode='rgb_array')
    # env = Monitor(env, filename='/Users/rhyscooper/Desktop/MScProject/Pages/logs/starbs')
    env.OPPONENT.policy = 'random'
    env.AGENT.policy = 'PPO'
    
    gen_keys = ['1']
    storage = metric_dicts()
    storage.add_keys_to_metrics(gen_keys)

    # env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn = nn.ReLU, net_arch={'pi':[64,64], 'vf': [64,64]},  verbose=1, n_steps=64, batch_size= 32, n_epochs =1)
    # below threw an error in chonga hyps search
    # 'batch_size': 32, 'n_steps': 3072, 'learning_rate': 0.034440481970522255, 'n_epochs': 90.0, 'ent_coef': 0.0025, 'vf_coef': 0.25, 'activation_fn': <class 'torch.nn.modules.activation.Tanh'>, 'optimizer_class': <class 'torch.optim.adam.Adam'>, 'net_arch': {'pi': [256], 'vf': [256]}}
    env.AGENT.model =  PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.Tanh ,net_arch=  {'pi': [256], 'vf': [256]},learning_rate=  0.034440481970522255, n_steps = 3072, batch_size = 32, n_epochs=  90, ent_coef=  0.0025, vf_coef=  0.25)

    callback_train = CustomLoggerCallback() 
    env.AGENT.model.learn(total_timesteps=learning_steps, progress_bar=True, callback= callback_train, dumb_mode=False)
    storage.update_train_metrics_from_callback('1', callback_train)
    
    
    
    eval_env = texas_holdem.env('72+', render_mode='rgb_array')
    eval_env = Monitor(eval_env)
    eval_env.AGENT.policy = 'PPO'
    eval_env.OPPONENT.policy = 'random'

    mean_reward,episode_rewards, episode_lengths= evaluate_policy(eval_env.AGENT.model, eval_env, callback=None, return_episode_rewards =True, n_eval_episodes = self.n_eval_episodes)
            
    metric_dicts.gen_eval_rewards[gen_keys[0]] = episode_rewards
    percentages, p = extract_opt_act(eval_env)
    metric_dicts.update_eval_metrics_from_ep_rewards(gen_keys[0], mean_reward,episode_rewards, percentages)
    

    
    gm = graph_metrics(n_models = 2, storage = storage, storageB = None, figsize= (10, 8), t_steps = learning_steps, overlay = False )
    gm.print_all_graphs(True, False)


# print(simp_PPO(50000))    

def simp_PPO2():
    env = texas_holdem.env()
    # env = Monitor(env, filename='/Users/rhyscooper/Desktop/MScProject/Pages/logs/starbs')
    env.OPPONENT.policy = 'random'
    env.AGENT.policy = 'DQN'

    # env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn = nn.ReLU, net_arch={'pi':[64,64], 'vf': [64,64]},  verbose=1, n_steps=64, batch_size= 32, n_epochs =1)
    # below threw an error in chonga hyps search
    env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.SGD,activation_fn= nn.ReLU,net_arch=  {'pi': [64, 64], 'vf': [64, 64]},learning_rate=  0.0060, n_steps = 2048, batch_size = 64, n_epochs=  100, ent_coef=  0.001, vf_coef=  0.85)
    env.AGENT.model.learn(total_timesteps=50000, progress_bar=True )
    # print(results)
    # env.AGENT.model.learn(total_timesteps=20, progress_bar=True)
    
    Eval_env = texas_holdem.env()
    Eval_env = Monitor(Eval_env)
    Eval_env.OPPONENT.policy = 'random'
    Eval_env.AGENT.policy = 'PPO'
        
    mean_reward, _ = evaluate_policy(env.AGENT.model, Eval_env, n_eval_episodes=1000)
    return mean_reward


def simp_A2C():
    env = texas_holdem.env()
    # env = Monitor(env, filename='/Users/rhyscooper/Desktop/MScProject/Pages/logs/starbs')
    env.OPPONENT.policy = 'random'
    env.AGENT.policy = 'A2C'

    # env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn = nn.ReLU, net_arch={'pi':[64,64], 'vf': [64,64]},  verbose=1, n_steps=64, batch_size= 32, n_epochs =1)
    # below threw an error in chonga hyps search
    env.AGENT.model = A2C('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn = nn.ReLU, net_arch=None, use_rms_prop = False)
    env.AGENT.model.learn(total_timesteps=500, progress_bar=True )
    # print(results)
    # env.AGENT.model.learn(total_timesteps=20, progress_bar=True)
    
    Eval_env = texas_holdem.env()
    Eval_env = Monitor(Eval_env)
    Eval_env.OPPONENT.policy = 'random'
    Eval_env.AGENT.policy = 'PPO'
        
    mean_reward, _ = evaluate_policy(env.AGENT.model, Eval_env, n_eval_episodes=1000)
    return mean_reward

# simp_A2C()
# print(simp_A2C)
    
class obs_experiment():
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
        self.trained_opponents = {}
        for key in self.train_envs.keys():
            sp = self_play(0, 20480, 1, obs_type = key, tag = key, model = self.model, na_key = self.na)
            sp.run(False)
            root = sp.gen_lib[gen_to_load]
            self.trained_opponents[key] = root
 
        
    def init_agent_opponent_models(self):   
        for id in self.train_envs.keys():
            env = self.train_envs[id]
            env.OPPONENT.policy = 'PPO'
            # env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam,activation_fn= nn.ReLU,net_arch=  {'pi': [256], 'vf': [256]},learning_rate=  0.07100101556878591, n_steps = 2048, batch_size = 128, n_epochs=  31)
            env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
            env.AGENT.policy = 'PPO'
            # env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam,activation_fn= nn.ReLU,net_arch=  {'pi': [256], 'vf': [256]},learning_rate=  0.07100101556878591, n_steps = 2048, batch_size = 128, n_epochs=  31)
            env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
            env.OPPONENT.model.set_parameters(load_path_or_dict= self.trained_opponents[id])
            
        for id in self.eval_envs.keys():
            env = self.eval_envs[id]
            env.OPPONENT.policy = 'PPO'
            # env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam,activation_fn= nn.ReLU,net_arch=  {'pi': [256], 'vf': [256]},learning_rate=  0.07100101556878591, n_steps = 2048, batch_size = 128, n_epochs=  31 )
            env.AGENT.policy = 'PPO'
            env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
            # env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam,activation_fn= nn.ReLU,net_arch=  {'pi': [256], 'vf': [256]},learning_rate=  0.07100101556878591, n_steps = 2048, batch_size = 128, n_epochs=  31)
            env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
            env.OPPONENT.model.set_parameters(self.trained_opponents[id])

        
                    
    def agent_train_eval(self):
        for id in self.train_envs.keys():  
            
            callback_train = CustomLoggerCallback()     
            env = self.train_envs[id]
            print(id, "learning")
            env.AGENT.model.learn(self.total_timesteps, dumb_mode = False,progress_bar=True, callback=callback_train)
            
            self.metric_dicts.update_train_metrics_from_callback(id, callback_train)
            
            rews = callback_train.rewards
            mean, max_val, min_val, data_range = list_statistics(rews)
            print("id", id)
            print("Mean:", mean)
            print("Max:", max_val)
            print("Min:", min_val)
            print("Range:", data_range)


            # occassionally rewards get stuck in a +- 0 band, below line reruns if this is the case. 
            if data_range < 20:
                print("invalid reward band")
                env.reset()
                env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
                env.AGENT.model.learn(self.total_timesteps, dumb_mode = False,progress_bar=True, callback=callback_train)
                self.metric_dicts.update_train_metrics_from_callback(id, callback_train) 
            else:   
                self.metric_dicts.update_train_metrics_from_callback(id, callback_train) 
                
            env.reset()
                
            callback_train_rand_op = CustomLoggerCallback()              
            env.AGENT.model.learn(total_timesteps = self.total_timesteps, dumb_mode = True, callback=callback_train_rand_op , progress_bar=True,)
            self.metric_dicts_rand.update_train_metrics_from_callback(id, callback_train_rand_op)
 
        for id in self.eval_envs.keys():
            env = self.eval_envs[id]
            env.AGENT.model = self.train_envs[id].AGENT.model
            
            mean_reward,episode_rewards, episode_lengths= evaluate_policy(env.AGENT.model, env, callback=None, return_episode_rewards =True, n_eval_episodes = self.n_eval_episodes)
            
            # self.metric_dicts.gen_eval_rewards[id] = episode_rewards
            
            self.metric_dicts.update_eval_metrics_from_ep_rewards(id, mean_reward,episode_rewards)
            
            callback_eval_rand_op = CustomLoggerCallback()
            env.AGENT.model.learn(total_timesteps = self.n_eval_episodes, dumb_mode = True, callback=callback_eval_rand_op , progress_bar=True,)
                
            mean_reward = callback_eval_rand_op.final_mean_reward
            episode_rewards = callback_eval_rand_op.rewards
            self.metric_dicts_rand.update_eval_metrics_from_ep_rewards(gen = id, mean_reward = mean_reward, episode_rewards = episode_rewards)


    def get_results(self, graphs):
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
#n eval episodes will always be at least n_epochs because running dumb agent to learn. 
                
    experiment = obs_experiment(total_timesteps =35000, n_eval_episodes = 2048, model = 'PPO')
    experiment.train_opponents(gen_to_load= 0)
    experiment.init_agent_opponent_models()
    print(experiment.agent_train_eval())
    experiment.get_results(graphs = True)


class obs_exp_exp():
    def __init__(self) -> None:
        self.results_dict = {}
        self.onetwofour = []
        self.seventytwo = []
        self.seventytwoplus = []
        
        
    def run(self):
        for i in range(1, 11):
            experiment = obs_experiment(total_timesteps =20480, n_eval_episodes = 2048, model = 'PPO')
            experiment.train_opponents(gen_to_load= 0)
            experiment.init_agent_opponent_models()
            print(experiment.agent_train_eval())
            self.results_dict[i] = experiment.get_results(graphs = False)
    def process(self):
        for key in self.results_dict.keys():
            scores = self.results_dict[key][0]
            self.onetwofour.append(scores['124'])
            self.seventytwo.append(scores['72'])
            self.seventytwoplus.append(scores['72+'])
                    
        self.onetwofour_mean = np.mean(self.onetwofour)
        self.seventytwo_mean = np.mean(self.seventytwo)
        self.seventytwoplus_mean  = np.mean(self.seventytwoplus)
                    
            
        
        return self.onetwofour_mean, self.seventytwo_mean,self.seventytwoplus_mean     
                
                
        
# obs_exp = obs_exp_exp()
# obs_exp.run()  
# print(obs_exp.process())   

# (-0.2940735, -0.227771, -0.22157471)

# sp_ppo = self_play(2, 100, 100, '72+', 12, 'PPO', na_key ={'pi': [64], 'vf': [64]} )
# sp_ppo.run(False)
# ppo_lib = sp_ppo.gen_lib


# sp_a2c = self_play(2, 100, 100, '72+', 12, 'A2C', na_key ={'pi': [64], 'vf': [64]} )
# sp_a2c.run(False)
# a2c_lib = sp_a2c.gen_lib


class PPO_vs_OPPONENT():
    def __init__(self, gen_lib_ppo, gen_to_load_ppo, gen_lib_a2c, gen_to_load_a2c, obs_type):
        self.gen_to_load_ppo = gen_to_load_ppo
        self.gen_lib_ppo = gen_lib_ppo
        self.gen_lib_a2c = gen_lib_a2c
        self.gen_to_load_a2c = gen_to_load_a2c
        self.obs_type = obs_type
        self.n_gens = 1
        
        self.storageA= metric_dicts(self.n_gens)
        self.storageB= metric_dicts(self.n_gens)
       

    def init_eval_env(self, opponent_type):
        self.eval_env = texas_holdem.env(self.obs_type, render_mode='rgb_array')
        self.eval_env = Monitor(self.eval_env)
        self.eval_env.AGENT.policy = 'PPO'
        self.eval_env.OPPONENT.policy = opponent_type
       
        
        #replace below w best model from hyptun
        # self.eval_env.AGENT.model = PPO('MultiInputPolicy', self.eval_env, optimizer_class = th.optim.Adam,activation_fn= nn.ReLU,net_arch=  {'pi': [64], 'vf': [64]},learning_rate=  0.07100101556878591, n_steps = 2048, batch_size = 128, n_epochs=  31, ent_coef=  0.000125, vf_coef=  0.25)
        self.eval_env.AGENT.model =  PPO('MultiInputPolicy', self.eval_env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = {'pi': [64], 'vf': [64]}, n_steps=10 )
        if self.eval_env.OPPONENT.policy == 'A2C':
            self.eval_env.OPPONENT.model = A2C('MultiInputPolicy', self.eval_env, optimizer_class = th.optim.Adam, activation_fn = nn.ReLU, net_arch=None, use_rms_prop = False)
        
        if self.eval_env.OPPONENT.policy == 'PPO':
            # self.eval_env.OPPONENT.model = PPO('MultiInputPolicy', self.eval_env, optimizer_class = th.optim.Adam,activation_fn= nn.ReLU,net_arch=  {'pi': [64], 'vf': [64]},learning_rate=  0.07100101556878591, n_steps = 2048, batch_size = 128, n_epochs=  31, ent_coef=  0.000125, vf_coef=  0.25)    
            self.eval_env.OPPONENT.model =  PPO('MultiInputPolicy', self.eval_env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = {'pi': [64], 'vf': [64]}, n_steps=10 )
    def load_params(self):
        root = self.gen_lib_ppo[self.gen_to_load_a2c]
        self.eval_env.AGENT.model.set_parameters(load_path_or_dict= root)
        
        root = self.gen_lib_a2c[self.gen_to_load_a2c]
        self.eval_env.AGENT.model.set_parameters(load_path_or_dict= root)
        self.eval_env.reset()
    
    def load_metric_dicts_storage(self):
        if self.eval_env.AGENT.policy ==self.eval_env.OPPONENT.policy:
            self.id_keys = [self.eval_env.AGENT.policy, self.eval_env.OPPONENT.policy + 'op']
        else:
            self.id_keys = [self.eval_env.AGENT.policy, self.eval_env.OPPONENT.policy]
        self.storageA.add_keys_to_metrics(self.id_keys)
        self.storageB.add_keys_to_metrics(self.id_keys)
    
    def evaluate(self, n_eval_episodes):
        self.metric_dicts = self.storageA
        self.metric_dicts_rand = self.storageB
        self.n_eval_episodes = n_eval_episodes
        
        #evaluate agent
        mean_reward,episode_rewards, episode_lengths= evaluate_policy(self.eval_env.AGENT.model, self.eval_env, callback=None, return_episode_rewards =True, n_eval_episodes = self.n_eval_episodes)
                
        self.metric_dicts.gen_eval_rewards[ self.id_keys[0]] = episode_rewards
        percentages, p = extract_opt_act(self.eval_env)
        self.metric_dicts.update_eval_metrics_from_ep_rewards( self.id_keys[0], mean_reward,episode_rewards, percentages)
        
        self.eval_env.reset()
        
        # evaluate opponent
        mean_reward,episode_rewards, episode_lengths= evaluate_policy(self.eval_env.OPPONENT.model, self.eval_env, callback=None, return_episode_rewards =True, n_eval_episodes = self.n_eval_episodes)
        self.metric_dicts.gen_eval_rewards[ self.id_keys[1]] = episode_rewards
        percentages, p = extract_opt_act(self.eval_env)
        self.metric_dicts.update_eval_metrics_from_ep_rewards( self.id_keys[1], mean_reward,episode_rewards, percentages)
        
        # evaluate randop                 
        callback_eval_rand_op = CustomLoggerCallback() 
        self.eval_env.AGENT.model.learn(total_timesteps = 2048, dumb_mode = False, callback=callback_eval_rand_op , progress_bar=True,)
        
        mean_reward = callback_eval_rand_op.final_mean_reward
        episode_rewards = callback_eval_rand_op.rewards
        self.metric_dicts_rand.update_eval_metrics_from_ep_rewards(gen = 1, mean_reward = mean_reward, episode_rewards = episode_rewards, percentages = percentages)            
                
    def get_results(self, graphs):
        print(self.metric_dicts.gen_eval_final_mean_reward)
        
        if graphs:
            gm = graph_metrics(n_models = 2, storage = self.metric_dicts, storageB= self.metric_dicts_rand, figsize=(6,8), t_steps = self.n_eval_episodes)
            gm.create_x_y()
            gm.print_all_graphs(False, True)
            # gm.plot_rewards(True, True)
            # gm.plot_moving_rewards(True, True)
            # gm.plot_moving_mean(True, True)
            # gm.plot_loss()
 
# sp = self_play(1, 2000, 10, '124', 3) 
# sp.run()

# gen_lib_ppo = sp.gen_lib
# gen_lib_a2c = sp.gen_lib

# def PPOvsA2C():                  
#     PPO_VS_A2C = PPO_vs_OPPONENT(ppo_lib, 1, a2c_lib, 1, '72+')
#     PPO_VS_A2C.init_eval_env('PPO')
#     PPO_VS_A2C.load_params()
#     PPO_VS_A2C.load_metric_dicts_storage()
#     PPO_VS_A2C.evaluate(10)
#     PPO_VS_A2C.get_results(True)

# PPOvsA2C()
class train_convergence_search():
    def __init__(self, verbose, obs_type):
        self.verbose = verbose
        self.obs_type = obs_type
        self.model = 'PPO'
        # sp = self_play(0, 20480, 1, obs_type = self.obs_type, tag = 10, model = self.model)
        # sp.run(False)
        # self.root = sp.gen_lib[0]

        self.trial_results = {}
        

    def init_trained_op(self, training_steps):
        self.na_gen_0_dict = {}
        # na = self.net_arch
        self.na = [{'pi': [64], 'vf': [64]}]
        for na_key in self.na:
            sp = self_play(0, training_steps, 1, obs_type = self.obs_type, tag = 19, model = self.model, na_key = na_key)
            sp.run(False)
            self.na_gen_0_dict[str(na_key)] = sp.gen_lib[0]
# Define the Optuna callback
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
        cb_params = self.optimize_cb_params(trial)
        
        env = texas_holdem.env(self.obs_type, render_mode = "rgb_array")
        env.OPPONENT.policy = 'PPO'
        env.AGENT.policy = 'PPO'
        # env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class= th.optim.Adam, activation_fn= nn.ReLU, net_arch = {'pi': [256], 'vf': [256]}, verbose=0, batch_size= 128, n_steps= 2048, vf_coef=0.25, ent_coef=0.000125, learning_rate= 0.07100101556878591)
        env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
        env.OPPONENT.model.set_parameters(self.na_gen_0_dict[str(self.na[0])])
        
        self.env = env
        # self.env.seed = 1
        
        Eval_env = texas_holdem.env(self.obs_type, render_mode = "rgb_array")
        Eval_env = Monitor(Eval_env)
        Eval_env.OPPONENT.policy = 'random'
        Eval_env.AGENT.policy = 'PPO'

        cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals= cb_params['max_no_improvement_evals'], min_evals=cb_params['min_evals'], verbose =1)
        cb = EvalCallback(Eval_env, eval_freq=10000, callback_after_eval=cb, verbose=0, n_eval_episodes= cb_params['max_no_improvement_evals']) 
           
        # model = PPO('MultiInputPolicy', self.env, optimizer_class= th.optim.Adam, activation_fn= nn.ReLU, net_arch = {'pi': [256], 'vf': [256]}, verbose=0, batch_size= 128, n_steps= 2048, vf_coef=0.25, ent_coef=0.000125, learning_rate= 0.07100101556878591)
        model = PPO('MultiInputPolicy', self.env, optimizer_class = th.optim.Adam, activation_fn= nn.ReLU, net_arch = self.na)
        model.learn(total_timesteps=40000, dumb_mode = False, progress_bar=self.verbose, callback= cb)
        # mean_reward, _, episode_lengths = evaluate_policy(model, Eval_env, n_eval_episodes=10000, return_episode_rewards = True)       
        
        return cb.callback.parent.best_mean_reward
            
    def run(self, print_graphs):
        if __name__ == '__main__':
            study = optuna.create_study(direction= 'maximize')
        try:
            # cbr = self.callback()
            study.optimize(self.optimize_cb, n_trials=8, n_jobs=1, show_progress_bar= True, callbacks = [self.callback])
        except KeyboardInterrupt:
            print('Interrupted by keyboard.')
        print(study.best_params)
        if print_graphs == True:
            optuna.visualization.plot_param_importances(study).show()
            optuna.visualization.plot_optimization_history(study).show() 
            # self.plot_dictionary()
    
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
        
        # Set labels and title
        plt.xlabel('number of steps no imporvement, minimum number of callbacks')
        plt.ylabel('Mean Values')
        plt.title('Number of steps vs mean reward')
        
        # Show the plot
        plt.show()





        
                                           
TCS = train_convergence_search(verbose=False, obs_type= '72+')
# TCS.init_trained_op()
# TCS.run(print_graphs= False)

# TCS.plot_dictionary()
# results = {'100': 2.1565, '400':3.1085, '100': 2.5275,  '100': 2.572, ' 50':1.8645, '100': 3.096, '200': 2.9375, '400': 2.732, '200': 2.809 , '400': 2.038, '300': 2.8335, '400': 3.207,'400': 3.1015, '400': 2.964, '400': 2.2365}
#  results = {'300': 3.1965, '300': 3.082, '500': 2.638 , '400': 2.918, '400': 2.783 , '400': 3.3025, '500':  2.843, '500': 2.9205, '600': 2.708, '600':2.9135 ,  '400':  2.9725 , '300': 3.1575}

results = {'1000,60': 2.4341, '1000,60': 1.8836,  '4000,75': 2.1669, '500,75': 2.07295, '500,75': 2.00825, '2000,75': 1.9293, '2000,75': 1.92875, '500,60': 2.2359, '2000,75': 1.7597, '2000,75': 1.5784, '1000,75': 2.3316, '500,60': 1.2743}

# results = {'500,75': 2.2255, '4000,75':1.64435, '4000,75':1.7584, '4000,75':2.29155, '500,75':2.1034, '4000,75': 1.7246,  }

# results = {'2000, 2': 2.087, '2000, 2': 1.79825, '1000, 2': 1.7505, '2000, 3': 1.6505, '1000, 3': 1.924, '2000, 3': 1.9945, '4000, 3': 1.985375, '1000, 3': 1.8265 , '2000, 3': 2.02875, '4000, 2': 1.5198, '1000, 3':2.0885, '2000, 3': 1.757, '2000, 2': 2.038, '2000, 2': 1.763, '2000, 3': 1.795}
TCS.plot_dictionary(results)
print(TCS.trial_results)

# best = 400

# class upload_checker():
#     def __init__(self):
#         self.model = ('PPO')


#     def train_opponents(self, gen_to_load):
#         self.trained_opponents = {}
#         obs_type = ('72+')
#         sp = self_play(0, 2048, 1, obs_type = obs_type, tag = 21, model = self.model)
#         sp.run(False)
#         root = sp.gen_lib[gen_to_load]
#         self.trained_opponents[0] = root
 
#     def get_trained_params(self):
#         path = self.trained_opponents[0]
#         loaded_model = PPO.load(path)
#         loaded_model.set_parameters(path)
            
# uc = upload_checker()      
# uc.train_opponents(0)
# uc.get_trained_params()
      
      

    
    # PPO vs       