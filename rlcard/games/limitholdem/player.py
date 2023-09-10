from enum import Enum
# from stable_baselines3 import DQN
# from stable_baselines3 import A2C
import numpy as np
import matplotlib.pyplot as plt
from rlcard.agents import DQNAgent
from rlcard.utils.utils import print_card
from treys import Evaluator
from treys import Card

class PlayerStatus(Enum):
    ALIVE = 0
    FOLDED = 1
    ALLIN = 2
    
def format_legal_actions(legal_actions):
    inputs = legal_actions
    for i in legal_actions:
        if i == 'call':
            inputs[inputs.index('call')] = 'call, (0)'
        
        if i == 'raise':
            inputs[inputs.index('raise')] = 'raise, (1)'
        
        if i == 'fold':
            inputs[inputs.index('fold')] = 'fold, (2)'
        
        if i == 'check':
            inputs[inputs.index('check')] = 'check, (3)'
    return inputs

def _print_state(state):
    ''' Print out the state

    Args:
        state (dict): A dictionary of the raw state
        action_record (list): A list of the each player's historical actions
    '''
    state = state['raw_obs']
    print('\n=============== Community Card ===============')
    print_card(state['public_cards'])
    print('===============   Your Hand    ===============')
    print_card(state['hand'])
    print('===============     Chips      ===============')
    print('Your chips:   ', end='')
    for _ in range(state['my_chips']):
        print('+', end='')
    print('')    
    print('Opponent chips:'  , end='')
    for _ in range(state['all_chips'][1]):
        print('+', end='')
    print('\n=========== Actions You Can Choose ===========')
    
    print(format_legal_actions(state['legal_actions']))
  



class LimitHoldemPlayer:

    def __init__(self, player_id, np_random, policy):
        """
        Initialize a player.

        Args:
            player_id (int): The id of the player
        """
        self.np_random = np_random
        # self.player_id = player_id
        self.player_id = f"player_{player_id}"
        self.hand = []
        self.status = PlayerStatus.ALIVE
        self.policy = policy 
        self.model = None
        self.env = None
        self.rewardz = []
        self.opt_acts = []
        self.opt_actions2 = 0
        
            

        # The chips that this player has put in until now
        self.in_chips = 0

    def get_state(self, public_cards, all_chips, legal_actions):
        """
        Encode the state for the player

        Args:
            public_cards (list): A list of public cards that seen by all the players
            all_chips (int): The chips that all players have put in

        Returns:
            (dict): The state of the player
        """
        return {
            'hand': [c.get_index() for c in self.hand],
            'public_cards': [c.get_index() for c in public_cards],
            'all_chips': all_chips,
            'my_chips': self.in_chips,
            'legal_actions': legal_actions
        }

    def get_player_id(self):
        return self.player_id
    
    def apply_mask(initial_list, mask):
        masked_list = [value for value, valid in zip(initial_list, mask) if valid]
        return masked_list

    def get_action_mask(self):
        env = self.env
        obs = env.observe(self.player_id)
        if len(obs) > 2:
            mask = obs[0]['action_mask']
        else:
            mask = obs['action_mask'] 
        
        return mask
 
    def get_action(self, player_obs):
        # print("player obs", player_obs)    

        if self.policy == 'random':
            mask1 = player_obs['action_mask']
            # print("opponent action mask", mask1)
            action = self.env.action_space(self.player_id).sample(mask1)
            
        if self.policy == 'DQN':
            print("getting action for",(player_obs))
            if type(player_obs) == list:
                player_obs = player_obs[0]
            # if type(player_obs) == dict:
            #     player_obs = player_obs['observation']
            print("player_obs", player_obs,)
            action = self.model.predict(observation=player_obs)
            action = action[0]
            # self.model.policy.forward(player_obs)  
            
        
    
        if self.policy == 'PPO':
            if type(player_obs) == tuple:
                # print("obstypelist")
                player_obs = player_obs[0]
                # del player_obs['action_mask']
                # player_obs = player_obs['observation']
                pass
            # if type(player_obs) == dict:
            #     print("obstypedict")
            #     player_obs = player_obs['observation']
            # print("player_obs", player_obs,)
            action = self.model.predict(observation=player_obs)
            action = action[0]
            # self.optimal_action(action)
            # self.model.policy.forward(player_obs) 
            
        if self.policy == 'A2C':
            if type(player_obs) == tuple:
                player_obs = player_obs[0]
                pass
            action = self.model.predict(observation=player_obs)
            action = action[0]
            
                    
        if self.policy == 'human':
            raw_env = self.env.env.env.env
            # state = raw_env.get_state('player_0')
            lhm_env = self.env.env.env.env.env
            state  = lhm_env.get_state(0)
            print(_print_state(state))
            
            try:
                action = int(input('>> You choose action (integer): '))
            except ValueError:
                action = int(input('>> You choose action (integer): '))
                
            action = action    
            
            
        # self.optimal_action(action)   
        # print(self.opt_acts) 
        return action 
                          
                          
        # return action 
    # def init_policy_model(self, env):
    #     if self.policy == 'DQN':
    #         self.model = DQN("MlpPolicy", env, verbose=1)
    #         self.model._setup_model()
    #         params = (self.model.policy)
    #         # print("params", params)
    #         # print("shape", (np.shape(params)))
    #         # self.model = A2C("MlpPolicy", env, device="cpu")
    # def player_learn(self):
    #     self.model.learn(total_timesteps=10, log_interval=4)
        
    # def player_train(self):
    #     self.model.train(gradient_steps=100)  
    
    # def optimal_action(self, action):
        
    #     score_max = 7462
    #     quartiles = [score_max * 0.25, score_max * 0.5, score_max * 0.75]
        
    #     game = self.env.env.env.env.env.game
        
    #     hand = []
    #     for c in self.hand:
    #         c1r = c.rank
    #         c1s = c.suit.lower()
    #         c1 = c1r +  c1s
    #         hand.append(c1)
  
    #     pc = []
    #     if len(game.public_cards) > 0:
    #         public_cards = game.public_cards
                     
    #         for c in public_cards:
    #             cr_temp = c.rank
    #             cs_temp = c.suit.lower()
    #             pc.append(cr_temp +  cs_temp)
                
    #     hand_objs = []
    #     pc_objs = [] 
    #     for c in hand:
    #         hand_objs.append(Card.new(c))   
    #     for c in pc:
    #         pc_objs.append(Card.new(c))       
        
    #     if len(pc) >= 3:
    #         evaluator = Evaluator()
    #         try: 
    #             score = evaluator.evaluate(hand_objs, pc_objs)
    #         except:
    #             KeyError
    #             score = 0
            
    #         if score <= quartiles[0]:
    #             op_act = 3
    #         if score >= quartiles[0] and score <= quartiles[1]:
    #             op_act = 4
    #         if score >= quartiles[1] and score <= quartiles[2]:
    #             op_act = 0
    #         if score >= quartiles[2]:
    #             op_act = 1   
    #         if action == op_act:
    #             self.opt_acts.append(1) 
    #             self.opt_actions2 +=1       
    #         else:  
    #             self.opt_acts.append(0)         
            
               
        
                 
                
        

