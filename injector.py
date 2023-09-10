
import numpy as np   
import texas_holdem_mod as texas_holdem
import rlcard 
from rlcard.utils.utils import print_card as prnt_cd
from rlcard.games.base import Card
import rlcard 
from rlcard.utils.utils import print_card as prnt_cd
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from callbacks_mod import EvalCallback
from callbacks_mod import StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from callbacks_mod import BaseCallback
import os 
import matplotlib.pyplot as plt
from ppo import PPO
from a2c import A2C
from stable_baselines3 import DQN
from gymnasium import Env
import optuna
import gym
import numpy as np
import torch as th
from torch import nn
from tabulate import tabulate
import pandas as pd
from rlcard.agents.human_agents.nolimit_holdem_human_agent import HumanAgent
from classmaker import graph_metrics
from classmaker import obs_type_envs
from classmaker import metric_dicts

import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

env = texas_holdem.env('72+', render_mode = "rgb_array")
env.AGENT.policy = 'PPO'
env.OPPONENT.policy = 'PPO'
env.OPPONENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam,activation_fn= nn.ReLU,net_arch=  {'pi': [256], 'vf': [256]},learning_rate=  0.07100101556878591, n_steps = 100, batch_size = 50, n_epochs=  31, ent_coef=  0.000125, vf_coef=  0.25)
env.AGENT.model = PPO('MultiInputPolicy', env, optimizer_class = th.optim.Adam,activation_fn= nn.ReLU,net_arch=  {'pi': [256], 'vf': [256]},learning_rate=  0.07100101556878591, n_steps = 100, batch_size = 50, n_epochs=  31, ent_coef=  0.000125, vf_coef=  0.25)
env.AGENT.model.learn(10, False, None, progress_bar=True)
env.OPPONENT.model.learn(10, False, None, progress_bar=True)


class card_injector():
    def __init__(self, agent, opponent, env):
        self.agent = agent
        self.opponent = opponent
        self.env = env
        

        self.hand_HC= []
        self.com_cards_HC = []
        
        self.hand_STR= []
        self.com_cards_STR = []
        
        self.hand_RF= []
        self.com_cards_RF = []
        
        self.high_card = {'hand': [('S', 'A'), ('D', 'T')], 'com_cards':[('H','8'), ('C','4'), ('D', '7'), ('S', '2'), ('H', 'K')]}
        self.straight = {'com_cards':[('H','6'), ('D', '7'), ('S', '8'), ('C','9'), ('H','T')], 'hand':[('D', 'J'), ('C','Q')]}
        self.royal_flush = {'hand':[('H', 'A'), ('H', 'K')], 'com_cards':[('S', 'T'), ('S', 'Q'), ('S', 'K'), ('S', 'A'), ('S', 'J')]}
        
        self.title_list = ['high_card', 'straight', 'royal_flush']
        
        with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)
        
        for title in self.title_list:
            self.premaker(title)
        
        self.pre_made_dict = {'HC': (self.hand_HC, self.com_cards_HC), 'STR':(self.hand_STR, self.com_cards_STR), 'RF': (self.hand_RF, self.com_cards_RF)}    
        self.obs_dict = {}
        
        self.create_obs()    
        
        self.get_probs_similarity()    
    def premaker(self, title):    
        if title == 'high_card':

            card_list = self.high_card['hand']
            for c in card_list:
                self.hand_HC.append(Card(*c))
            card_list = self.high_card['com_cards']
            for c in card_list:
                self.com_cards_HC.append(Card(*c))    
            
        if title == 'straight':
            card_list = self.straight['hand']
            for c in card_list:
                self.hand_STR.append(Card(*c))
            card_list = self.straight['com_cards']
            for c in card_list:
                self.com_cards_STR.append(Card(*c))    
            
        
        if title == 'royal_flush':
            card_list = self.royal_flush['hand']
            for c in card_list:
                self.hand_RF.append(Card(*c))
            card_list = self.royal_flush['com_cards']
            for c in card_list:
                self.com_cards_RF.append(Card(*c))     
                  
    def create_cards_hand(self, card_list):
        for c in card_list:
            self.hand.append(Card(*c))
    
    def create_cards_com(self, card_list):
        for c in card_list:
            self.com_cards.append(Card(*c)) 
            

    def create_obs(self):
        for key in self.pre_made_dict:
            tupe = self.pre_made_dict[key]
            extracted_state = {}
            if self.env.obs_type == '72+':
                hand_idx = [self.card2index[card.get_index()] for card in tupe[0]]
                public_cards_idx = [self.card2index[card.get_index()] for card in tupe[1]]
                obs = np.zeros(72)
                obs[public_cards_idx] = 1
                obs[hand_idx] = 2
                
            if key == 'HC':  
                raise_nums = [1,1,1]
                for i, num in enumerate(raise_nums):
                    obs[50 + i * 5 + num] = 1
                extracted_state['observation'] = obs
                extracted_state['action_mask'] = [1,0,1,1]
                
            if key == 'STR':  
                raise_nums = [2,3,4]
                for i, num in enumerate(raise_nums):
                    obs[50 + i * 5 + num] = 1
                extracted_state['observation'] = obs
                extracted_state['action_mask'] = [1,0,1,1]
                
            if key == 'RF':  
                raise_nums = [2,4,5]
                for i, num in enumerate(raise_nums):
                    obs[50 + i * 5 + num] = 1
                extracted_state['observation'] = obs     
                extracted_state['action_mask'] = [1,0,1,1]
            
            extracted_state['observation'] = np.array(extracted_state['observation'], dtype = np.float32)
            extracted_state['action_mask'] = np.array(extracted_state['action_mask'], dtype = np.int8)   
            dummy_info = {'reward': 0, 'done': False, 'truncation':False}
            self.obs_dict[key] = (extracted_state,dummy_info)
    
    def get_probs_similarity (self):
        self.sim_results = {}
        self.probs_results = {}
        for key in self.obs_dict:
            obs = self.obs_dict[key]
      
            obs = obs[0]
            dist_ag = (self.agent.model.policy.get_distribution(obs))
            probs_ag = dist_ag.distribution.probs
        
            
            self.create_obs()
            obs = self.obs_dict[key]
            obs = obs[0]
            dist_op = (self.opponent.model.policy.get_distribution(obs))
            probs_op = dist_op.distribution.probs
            
            self.probs_results[key] = [probs_ag, probs_op]
            self.sim_results[key] = self.calculate_cosine_similarity(probs_ag, probs_op)
    
    def calculate_cosine_similarity(self, list1, list2):
        vector1 = np.array(list1).reshape(1, -1)
        vector1 = np.delete(vector1, 1)
        vector1 = np.array(vector1).reshape(1, -1)
        
        vector2 = np.array(list2).reshape(1, -1)
        vector2 = np.delete(vector2, 1)
        vector2 = np.array(vector2).reshape(1, -1)

        
        similarity_matrix = cosine_similarity(vector1, vector2)
        similarity = similarity_matrix[0][0]
        
        return similarity

    def return_results(self):
        return (self.sim_results)

    
ci = card_injector(env.AGENT, env.OPPONENT, env)
# ci.premaker('high_card')
# print(ci.create_obs()) 
# sim = ci.get_probs_similarity()
# print(sim)