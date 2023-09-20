import json
import os
import numpy as np
from collections import OrderedDict

import rlcard
from rlcard.envs import Env
from rlcard.games.limitholdem import Game

DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        }

class LimitholdemEnv(Env):
    ''' Limitholdem Environment
    '''

    def __init__(self, config):
        ''' Initialize the Limitholdem environment
        '''
        self.name = 'limit-holdem'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.actions = ['call', 'raise', 'fold', 'check']
        self.obs_shape = None
        self.chips_type = None
        self.action_shape = [None for _ in range(self.num_players)]

        with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)

    def set_state_shape(self, obs_shape, obs_type):
        """
        set the state shape, ie the boolean vector length depending on the observation type. 

        Args:
            obs_shape (_type_): boolean vector length
            obs_type (_type_): observation type (124 or 72 or 72+ or PIG)
        """
        self.obs_shape = obs_shape 
        self.obs_type = obs_type
        # set the state shape, ie the boolean vector length depending on the observation type.       
        if self.obs_shape[0] == '124' and self.obs_type == '124':
            self.state_shape = [[124] for _ in range(self.num_players)]
        
        if self.obs_shape[0] == '72' and self.obs_type == '72':
            self.state_shape = [[72] for _ in range(self.num_players)]
            
        if self.obs_shape[0] == '72' and self.obs_type == '72+':
            self.state_shape = [[72] for _ in range(self.num_players)]
        
        if self.obs_shape[0] == '124' and self.obs_type == 'PIG':
            self.state_shape = [[124] for _ in range(self.num_players)]    
             
    def set_chips_type(self, chips_type):
        self.chips_type = chips_type         
                   
    def _get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def _extract_state(self, state):
        '''
        Extracts the state representation from the state dictionary for the agent.

        Args:
            state (dict): Original state from the game.

        Returns:
            extracted_state (dict): A dictionary containing the extracted state representation.
                - 'legal_actions' (OrderedDict): Legal actions available to the agent.
                - 'obs' (numpy.ndarray): The observation array based on the specified observation shape and type.
                - 'raw_obs' (dict): The raw state observation from the game.
                - 'raw_legal_actions' (list): List of raw legal actions.
                - 'action_record' (ActionRecorder): The action recorder for the agent.
        '''
        extracted_state = {}

        legal_actions = OrderedDict({self.actions.index(a): None for a in state['legal_actions']})
        extracted_state['legal_actions'] = legal_actions
        all_chips = state['all_chips']
        public_cards = state['public_cards']
        hand = state['hand']
        raise_nums = state['raise_nums']
        
        # first deck of cards indexes the player cards, second indexes the community cards.
        if self.obs_shape[0] == '124' and self.obs_type == '124':
            hand_idx = [self.card2index[card] for card in hand]
            public_cards_idx = [self.card2index[card] for card in public_cards]
            obs = np.zeros(124)
            obs[hand_idx] = 1
            obs[:52][public_cards_idx] = 2

            for i, num in enumerate(raise_nums):
                obs[100 + i * 5 + num] = 1
            extracted_state['obs'] = obs
            extracted_state['raw_obs'] = state
            extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
            extracted_state['action_record'] = self.action_recorder
            

        # default RLCard observation type
        if self.obs_shape[0] == '72' and self.obs_type == '72':
            cards = public_cards + hand
            idx = [self.card2index[card] for card in cards]
            obs = np.zeros(72)
            obs[idx] = 1

            for i, num in enumerate(raise_nums):
                obs[50 + i * 5 + num] = 1
            extracted_state['obs'] = obs
            extracted_state['raw_obs'] = state
            extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
            extracted_state['action_record'] = self.action_recorder
            
        # for 72+ observation type, public cards have index 1, player cards have index 2. opponent cards are unseen  
        if self.obs_shape[0] == '72' and self.obs_type == '72+':
            hand_idx = [self.card2index[card] for card in hand]
            public_cards_idx = [self.card2index[card] for card in public_cards]
            obs = np.zeros(72)
            obs[public_cards_idx] = 1
            obs[hand_idx] = 2

            for i, num in enumerate(raise_nums):
                obs[50 + i * 5 + num] = 1
            extracted_state['obs'] = obs
            extracted_state['raw_obs'] = state
            extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
            extracted_state['action_record'] = self.action_recorder    
        
        # for Perfect information game (PIG), oppponents cards are included in the observation.
        if self.obs_shape[0] == '124' and self.obs_type == 'PIG':
            hand_idx = [self.card2index[card] for card in hand]
            public_cards_idx = [self.card2index[card] for card in public_cards]
            
            op_state = self.game.get_state(player= 0)
            op_cards = op_state['hand']
            op_card_idx = [self.card2index[card] for card in op_cards ]
            obs = np.zeros(124)
            obs[public_cards_idx] = 1
            obs[hand_idx] = 2
            obs[:52][op_card_idx] =3
            
            for i, num in enumerate(raise_nums):
                obs[100 + i * 5 + num] = 1
            extracted_state['obs'] = obs
            extracted_state['raw_obs'] = state
            extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
            extracted_state['action_record'] = self.action_recorder       
            
        return extracted_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        return self.game.get_payoffs()

    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        legal_actions = self.game.get_legal_actions()
        if self.actions[action_id] not in legal_actions:
            if 'check' in legal_actions:
                return 'check'
            else:
                return 'fold'
        return self.actions[action_id]

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['chips'] = [self.game.players[i].in_chips for i in range(self.num_players)]
        state['public_card'] = [c.get_index() for c in self.game.public_cards] if self.game.public_cards else None
        state['hand_cards'] = [[c.get_index() for c in self.game.players[i].hand] for i in range(self.num_players)]
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        return state
