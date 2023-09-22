
import numpy as np   
import texas_holdem_mod as texas_holdem
import rlcard 
from rlcard.utils.utils import print_card as prnt_cd
from rlcard.games.base import Card
import rlcard 
from rlcard.utils.utils import print_card as prnt_cd
try: from stable_baselines3.common.env_checker import check_env
except ModuleNotFoundError: from env_checker_mod import check_env

try: from stable_baselines3.common.evaluation import evaluate_policy
except ModuleNotFoundError: from evaluation_mod import evaluate_policy
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
from classmaker import metric_dicts

import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.special import kl_div




class card_injector():
    """
    A class for analyzing and comparing probability distributions of card combination observations.

    This class takes an agent and an opponent, injects specific card combinations, and calculates the KL divergence
    or cosine similarity between the resulting probability distributions of actions taken by the agent and opponent.

    Parameters:
    - agent: The agent model to evaluate.
    - opponent: The opponent model to evaluate.
    - env: The Texas Hold'em environment.

    Attributes:
    - agent: The agent model to evaluate.
    - opponent: The opponent model to evaluate.
    - env: The Texas Hold'em environment.
    - hand_HC: A list of high card hand combinations.
    - com_cards_HC: A list of common cards for high card combinations.
    - hand_STR: A list of straight hand combinations.
    - com_cards_STR: A list of common cards for straight combinations.
    - hand_RF: A list of royal flush hand combinations.
    - com_cards_RF: A list of common cards for royal flush combinations.
    - hand_PR: A list of pair hand combinations.
    - com_cards_PR: A list of common cards for pair combinations.
    - hand_FLSH: A list of flush hand combinations.
    - com_cards_FLSH: A list of common cards for flush combinations.
    - hand_STR_FLSH: A list of straight flush hand combinations.
    - com_cards_STR_FLSH: A list of common cards for straight flush combinations.
    - high_card: A predefined high card combination.
    - pair: A predefined pair combination.
    - straight: A predefined straight combination.
    - flush: A predefined flush combination.
    - straight_flush: A predefined straight flush combination.
    - royal_flush: A predefined royal flush combination.
    - title_list: A list of predefined combination titles.
    - card2index: A dictionary mapping card names to their indices.
    - pre_made_dict: A dictionary containing pre-made combinations.
    - obs_dict: A dictionary containing observations for each combination.
    - div_results: A dictionary to store KL divergence results for each combination.
    - sim_results: A dictionary to store cosine similarity results for each combination.

    Methods:
    - premaker(title): Creates predefined card combinations based on the specified title.
    - create_obs(): Creates observations for each combination and stores them in obs_dict.
    - get_kl_div(): Calculates KL divergence between agent and opponent probabilities for each combination.
    - get_probs_similarity(): Calculates cosine similarity between agent and opponent probabilities for each combination.
    - calculate_cosine_similarity(list1, list2): Calculates cosine similarity between two lists.
    - calculate_kl_divergence(list1, list2, smoothing_alpha): Calculates KL divergence between two lists.
    - return_results(): Returns the KL divergence results.

    """
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

        self.hand_PR = []
        self.com_cards_PR = []

        self.hand_FLSH = []
        self.com_cards_FLSH = []

        self.hand_STR_FLSH = []
        self.com_cards_STR_FLSH =[]
        
        
        self.high_card = {'hand': [('S', 'A'), ('D', 'T')], 'com_cards':[('H','8'), ('C','4'), ('D', '7'), ('S', '2'), ('H', 'K')]}
        self.pair = {'hand': [('S', 'A'), ('D', 'A')], 'com_cards':[('H','8'), ('C','4'), ('D', '7'), ('S', '2'), ('H', 'K')]}
        self.straight = {'com_cards':[('H','6'), ('D', '7'), ('S', '8'), ('C','9'), ('H','T')], 'hand':[('D', 'J'), ('C','Q')]}
        self.flush = {'hand': [('D', 'A'), ('D', 'T')], 'com_cards':[('D','6'), ('D','2'), ('D','3'), ('D','5'), ('D','7')]}
        self.straight_flush = {'hand': [('D', '6'), ('D', '7')], 'com_cards':[('D', '8'), ('D', '9'), ('D', 'T'), ('D', 'J'), ('D', 'Q')]}
        self.royal_flush = {'hand':[('H', 'A'), ('H', 'K')], 'com_cards':[('S', 'T'), ('S', 'Q'), ('S', 'K'), ('S', 'A'), ('S', 'J')]}
        
        self.title_list = ['high_card', 'pair', 'straight', 'flush', 'straight_flush', 'royal_flush' ]
        
        with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)
        
        for title in self.title_list:
            self.premaker(title)
        
        self.pre_made_dict = {'HC': (self.hand_HC, self.com_cards_HC), 'STR':(self.hand_STR, self.com_cards_STR), 'RF': (self.hand_RF, self.com_cards_RF), 'PR': (self.hand_PR, self.com_cards_PR), 'FLSH': (self.hand_FLSH, self.com_cards_FLSH), 'STR_FLSH': (self.hand_STR_FLSH, self.com_cards_STR_FLSH)}    
        self.obs_dict = {}
        
        self.create_obs()    
        
        self.get_kl_div()

    def premaker(self, title):    
        if title == 'pair':        
            card_list = self.pair['com_cards']
            for c in card_list:
                self.com_cards_PR.append(Card(*c))
            
            card_list = self.pair['hand']
            for c in card_list:
                self.hand_PR.append(Card(*c)) 

        if title == 'flush':        
            card_list = self.pair['com_cards']
            for c in card_list:
                self.com_cards_FLSH.append(Card(*c))
            
            card_list = self.pair['hand']
            for c in card_list:
                self.hand_FLSH.append(Card(*c))

        if title == 'straight_flush':        
            card_list = self.pair['com_cards']
            for c in card_list:
                self.com_cards_STR_FLSH.append(Card(*c))
            
            card_list = self.pair['hand']
            for c in card_list:
                self.hand_STR_FLSH.append(Card(*c))   

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
                
            if self.env.obs_type == '72':
                hand_idx = [self.card2index[card.get_index()] for card in tupe[0]]
                public_cards_idx = [self.card2index[card.get_index()] for card in tupe[1]]
                obs = np.zeros(72)
                obs[public_cards_idx] = 1
                obs[hand_idx] = 1

            if self.env.obs_type == 'PIG':
                hand_idx = [self.card2index[card.get_index()] for card in tupe[0]]
                public_cards_idx = [self.card2index[card.get_index()] for card in tupe[1]]
                op_state = self.env.game.get_state(player= 0)
                op_cards = op_state['hand']
                op_card_idx = [self.card2index[card] for card in op_cards ]
                obs = np.zeros(124)
                obs[public_cards_idx] = 1
                obs[hand_idx] = 2
                obs[:52][op_card_idx] =3
                    
            if self.env.obs_type != 'PIG':
                raise_nums_start_index = 50
            if self.env.obs_type == 'PIG':
                raise_nums_start_index = 100

            if key == 'HC':  
                raise_nums = [1,1,1]
                for i, num in enumerate(raise_nums):
                    obs[raise_nums_start_index  + i * 5 + num] = 1
                extracted_state['observation'] = obs
                extracted_state['action_mask'] = [1,0,1,1]
                
            if key == 'STR':  
                raise_nums = [2,3,4]
                for i, num in enumerate(raise_nums):
                    obs[raise_nums_start_index  + i * 5 + num] = 1
                extracted_state['observation'] = obs
                extracted_state['action_mask'] = [1,0,1,1]
                
            if key == 'RF':  
                raise_nums = [2,4,5]
                for i, num in enumerate(raise_nums):
                    obs[raise_nums_start_index  + i * 5 + num] = 1
                extracted_state['observation'] = obs     
                extracted_state['action_mask'] = [1,0,1,1]

            if key == 'STR_FLSH':  
                raise_nums = [2,4,5]
                for i, num in enumerate(raise_nums):
                    obs[raise_nums_start_index  + i * 5 + num] = 1
                extracted_state['observation'] = obs     
                extracted_state['action_mask'] = [1,0,1,1]    

            if key == 'FLSH':  
                raise_nums = [2,3,4]
                for i, num in enumerate(raise_nums):
                    obs[raise_nums_start_index  + i * 5 + num] = 1
                extracted_state['observation'] = obs     
                extracted_state['action_mask'] = [1,0,1,1]

            if key == 'PR':  
                raise_nums = [1,2,1]
                for i, num in enumerate(raise_nums):
                    obs[raise_nums_start_index  + i * 5 + num] = 1
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
    
    def get_kl_div(self):
        self.div_results = {}
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
            self.div_results[key] = self.calculate_kl_divergence(probs_ag, probs_op)        

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

    def calculate_kl_divergence(self, list1, list2, smoothing_alpha=0.00000000001):
        vector1 = np.array(list1).reshape(1, -1)
        # vector1 = np.delete(vector1, 1)
        # vector1 = np.array(vector1).reshape(1, -1)

        vector2 = np.array(list2).reshape(1, -1)
        # vector2 = np.delete(vector2, 1)
        # vector2 = np.array(vector2).reshape(1, -1)

        vector1_smoothed = (vector1 + smoothing_alpha) / (np.sum(vector1) + len(vector1[0]) * smoothing_alpha)
        vector2_smoothed = (vector2 + smoothing_alpha) / (np.sum(vector2) + len(vector2[0]) * smoothing_alpha)

        kl_v1_v2 = kl_div(vector1_smoothed ,  vector2_smoothed).sum()    
        return kl_v1_v2

    def return_results(self):
        return (self.div_results)

    
