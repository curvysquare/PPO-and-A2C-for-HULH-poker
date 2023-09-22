from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from rlcard.agents import DQNAgent
from rlcard.utils.utils import print_card
from treys import Evaluator
from treys import Card
import random

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
    """
        Attributes:
        np_random (np.random.RandomState): The random number generator.
        player_id (str): The identifier for the player.
        hand (list): The player's hole cards.
        status (PlayerStatus): The player's status (ALIVE, FOLDED, etc.).
        policy (str): The player's policy (e.g., 'random', 'PPO', 'human', 'heuristic').
        model: The player's machine learning model (if applicable).
        env: The game environment.
        rewardz (list): List to store rewards obtained during gameplay.
        opt_acts (list): List to store optimal actions taken by the player.
        in_chips (int): The chips that this player has put in the pot so far.

        Methods:
        get_state: Encode the state for the player
        get_player_id: Get the id of the player
        apply_mask: Apply a mask to a list
        get_action_mask: Get the action mask for the player
        get_action: Get the action for the player based off its policy attribute
    """

    def __init__(self, player_id, np_random, policy):
        """
        Initialize a player.

        Args:
            player_id (int): The id of the player
            np_random (np.random.RandomState): The random number generator
            policy (str): The policy of the player
        """
        self.np_random = np_random
        self.player_id = f"player_{player_id}"
        self.hand = []
        self.status = PlayerStatus.ALIVE
        self.policy = policy 
        self.model = None
        self.env = None
        self.rewardz = []
        self.opt_acts = []
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
 
    def get_action(self, player_obs, game):
        """
        Get the action to be taken by the player based on their policy.

        Args:
            player_obs (dict or tuple): The observation of the player, containing relevant game information.
            game: The game environment.

        Returns:
            action: The selected action according to the player's policy.  
        """
        if self.policy == 'random':
            mask1 = player_obs['action_mask']
            action = self.env.action_space(self.player_id).sample(mask1)

        if self.policy == 'PPO' or self.policy == 'A2C':
            if type(player_obs) == tuple:
                player_obs = player_obs[0]
                pass
            action = self.model.predict(observation=player_obs)
            action = action[0]
                       
        if self.policy == 'human':
            raw_env = self.env.env.env.env
            limit_holdem_env = self.env.env.env.env.env
            state  = limit_holdem_env.get_state(0)
            print(_print_state(state))
            try:
                action = int(input('>> You choose action (integer): '))
            except ValueError:
                action = int(input('>> You choose action (integer): '))
            action = action    
            
        if self.policy == 'heuristic':
           action = self.optimal_action(player_obs, game)
           if not self.check_for_one_at_index(player_obs['action_mask'], action):
            action = self.env.action_space(self.player_id).sample(player_obs['action_mask'])

        return action 

    def check_for_one_at_index(self, my_array, selected_index):
    # Check if the selected ction is valid
        if 0 <= selected_index < len(my_array):
            # Check if there is a 1 at the selected index
            if my_array[selected_index] == 1:
                return True
            else:
                return False
        else:
            return False                     
    def optimal_action(self, player_obs, game):
        """

        Determine the optimal action for the player 

        This method calculates theplayer's optimal action based on their current hand, the public cards in the game, and predefined quartile scores.
        If there are at least three public cards, it evaluates the hand's strength and selects an action based on quartile score ranges.
        If there are fewer than three public cards, it selects a random action from the available actions according to the provided action mask.

        Parameters:
        - player_obs (dict): A dictionary containing observation information for the AI player.
        - game (Game): The current poker game being played.

        Returns:
        - op_act (int): The selected optimal action for the AI player. Possible values:
            - 0: call
            - 1: raise 
            - 2: fold
            - 3: check
         """
        score_max = 7462
        quartiles = [score_max * 0.25, score_max * 0.5, score_max * 0.75]

        hand = []
        for c in self.hand:
            c1r = c.rank
            c1s = c.suit.lower()
            c1 = c1r +  c1s
            hand.append(c1)
  
        pc = []
        if len(game.public_cards) > 0:
            public_cards = game.public_cards
                     
            for c in public_cards:
                cr_temp = c.rank
                cs_temp = c.suit.lower()
                pc.append(cr_temp +  cs_temp)
                
        hand_objs = []
        pc_objs = [] 
        for c in hand:
            hand_objs.append(Card.new(c))   
        for c in pc:
            pc_objs.append(Card.new(c))       
        
        if len(pc) >= 3:
            evaluator = Evaluator()
            try: 
                score = evaluator.evaluate(hand_objs, pc_objs)
            except:
                KeyError
                score = 0
            
            if score <= quartiles[0]:
                op_act = 1
            if score >= quartiles[0] and score <= quartiles[1]:
                op_act = 0
            if score >= quartiles[1] and score <= quartiles[2]:
                op_act = 3
            if score >= quartiles[2]:
                op_act = 2  
        else:            
            mask1 = player_obs['action_mask']
            legal_acts =  [i for i, value in enumerate(mask1) if value == 1]
            op_act = random.choice(legal_acts)
      
            
        return op_act
  
               
        
                 
                
        

