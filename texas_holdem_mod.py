from pettingzoo.utils.wrappers.base import BaseWrapper
from gymnasium import Env
import matplotlib.pyplot as plt
from callbacks_mod import BaseCallback
from treys import Evaluator
from treys import Card



class CustomCallBack(BaseCallback):
    def __init__(self, verbose=0):
        super(BaseCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.reward_list = []

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
    


"""
# Texas Hold'em

```{figure} classic_texas_holdem.gif
:width: 140px
:name: texas_holdem
```

This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.classic import texas_holdem_v4` |
|--------------------|--------------------------------------------------|
| Actions            | Discrete                                         |
| Parallel API       | Yes                                              |
| Manual Control     | No                                               |
| Agents             | `agents= ['player_0', 'player_1']`               |
| Agents             | 2                                                |
| Action Shape       | Discrete(4)                                      |
| Action Values      | Discrete(4)                                      |
| Observation Shape  | (72,)                                            |
| Observation Values | [0, 1]                                           |


## Arguments

``` python
texas_holdem_v4.env(num_players=2)
```

`num_players`: Sets the number of players in the game. Minimum is 2.

### Observation Space

The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.

The main observation space is a vector of 72 boolean integers. The first 52 entries depict the current player's hand plus any community cards as follows

|  Index  | Description                                                 |
|:-------:|-------------------------------------------------------------|
|  0 - 12 | Spades<br>_`0`: A, `1`: 2, ..., `12`: K_                    |
| 13 - 25 | Hearts<br>_`13`: A, `14`: 2, ..., `25`: K_                  |
| 26 - 38 | Diamonds<br>_`26`: A, `27`: 2, ..., `38`: K_                |
| 39 - 51 | Clubs<br>_`39`: A, `40`: 2, ..., `51`: K_                   |
| 52 - 56 | Chips raised in Round 1<br>_`52`: 0, `53`: 1, ..., `56`: 4_ |
| 57 - 61 | Chips raised in Round 2<br>_`57`: 0, `58`: 1, ..., `61`: 4_ |
| 62 - 66 | Chips raised in Round 3<br>_`62`: 0, `63`: 1, ..., `66`: 4_ |
| 67 - 71 | Chips raised in Round 4<br>_`67`: 0, `68`: 1, ..., `71`: 4_ |

#### Legal Actions Mask

The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.

### Action Space

| Action ID | Action |
|:---------:|--------|
|     0     | Call   |
|     1     | Raise  |
|     2     | Fold   |
|     3     | Check  |

### Rewards

| Winner          | Loser           |
| :-------------: | :-------------: |
| +raised chips/2 | -raised chips/2 |

### Version History

* v4: Upgrade to RLCard 1.0.3 (1.11.0)
* v3: Fixed bug in arbitrary calls to observe() (1.8.0)
* v2: Bumped RLCard version, bug fixes, legal action mask in observation replaced illegal move list in infos (1.5.0)
* v1: Bumped RLCard version, fixed observation space, adopted new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
* v0: Initial versions release (1.0.0)

"""

import os

import gymnasium
import numpy as npx
import pygame

from rlcard_base_mod import RLCardBase
from pettingzoo.utils import wrappers

import numpy as np 
# Pixel art from Mariia Khmelnytska (https://www.123rf.com/photo_104453049_stock-vector-pixel-art-playing-cards-standart-deck-vector-set.html)


def get_image(path):
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    return image


def get_font(path, size):
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    font = pygame.font.Font((cwd + "/" + path), size)
    return font




class raw_env(RLCardBase):
    """
    A custom RL environment for Texas Hold'em poker. supports
    different observation types and rendering modes.

    Args:
        num_players (int): The number of players in the game.
        render_mode (str): The rendering mode to use ("human" for human-readable output, "rgb_array" for image-based
            rendering).
        obs_type (str): The observation type to use. Choose from '72' (for 72-dimensional observation space),
            '72+' (also 72-dimensional, but with seperate indexing for players cards), '124' (for 124-dimensional observation space, 52 index positions
            for players cards, another 52 index positions for community cards),
            or 'PIG' (for 124-dimensional observation space with a perfect infromation game format that includes opponents cards).

    Attributes:
        metadata (dict): Metadata describing the environment, including render modes, name, parallelizability, and
            rendering frames per second.
        obs_shape (str): The shape of the observation space ('72' or '124') based on the chosen obs_type.
        obs_type (str): The selected observation type ('72', '72+', '124', or 'PIG').
        render_mode (str): The rendering mode chosen for this environment ('human' or 'rgb_array').

    Methods:
        step(action): Perform one step of the environment given an action.
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "texas_holdem_v4",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, num_players, render_mode, obs_type):
        # Initialize observation shape and type based on the provided parameters
        if obs_type == '72': 
            self.obs_shape = '72'
            self.obs_type = obs_type
        elif obs_type == '72+': 
            self.obs_shape = '72'
            self.obs_type = obs_type
        elif obs_type == '124': 
            self.obs_shape = '124'
            self.obs_type = obs_type  
        elif obs_type == 'PIG':
            self.obs_shape = '124'
            self.obs_type = obs_type  
                
        # Initialize the base class with appropriate parameters
        super().__init__("limit-holdem_mod", num_players, (self.obs_shape,), self.obs_type)
        
        # Set the rendering mode
        self.render_mode = render_mode

    def step(self, action):
        """
        Perform one step of the environment.

        Args:
            action: The action to be taken in the environment.

        Returns:
            None
        """
        super().step(action)

        # Render the environment if the render mode is "human"
        if self.render_mode == "human":
            self.render()

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        def calculate_width(self, screen_width, i):
            return int(
                (
                    screen_width
                    / (np.ceil(len(self.possible_agents) / 2) + 1)
                    * np.ceil((i + 1) / 2)
                )
                + (tile_size * 31 / 616)
            )

        def calculate_offset(hand, j, tile_size):
            return int(
                (len(hand) * (tile_size * 23 / 56)) - ((j) * (tile_size * 23 / 28))
            )

        def calculate_height(screen_height, divisor, multiplier, tile_size, offset):
            return int(multiplier * screen_height / divisor + tile_size * offset)

        screen_height = 1000
        screen_width = int(
            screen_height * (1 / 20)
            + np.ceil(len(self.possible_agents) / 2) * (screen_height * 1 / 2)
        )

        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.event.get()
        elif self.screen is None:
            pygame.font.init()
            self.screen = pygame.Surface((screen_width, screen_height))

        # Setup dimensions for card size and setup for colors
        tile_size = screen_height * 2 / 10

        bg_color = (7, 99, 36)
        white = (255, 255, 255)
        self.screen.fill(bg_color)

        chips = {
            0: {"value": 10000, "img": "ChipOrange.png", "number": 0},
            1: {"value": 5000, "img": "ChipPink.png", "number": 0},
            2: {"value": 1000, "img": "ChipYellow.png", "number": 0},
            3: {"value": 100, "img": "ChipBlack.png", "number": 0},
            4: {"value": 50, "img": "ChipBlue.png", "number": 0},
            5: {"value": 25, "img": "ChipGreen.png", "number": 0},
            6: {"value": 10, "img": "ChipLightBlue.png", "number": 0},
            7: {"value": 5, "img": "ChipRed.png", "number": 0},
            8: {"value": 1, "img": "ChipWhite.png", "number": 0},
        }

        # Load and blit all images for each card in each player's hand
        for i, player in enumerate(self.possible_agents):
            state = self.env.game.get_state(self._name_to_int(player))
            for j, card in enumerate(state["hand"]):
                # Load specified card
                card_img = get_image(os.path.join("img", card + ".png"))
                card_img = pygame.transform.scale(
                    card_img, (int(tile_size * (142 / 197)), int(tile_size))
                )
                # Players with even id go above public cards
                if i % 2 == 0:
                    self.screen.blit(
                        card_img,
                        (
                            (
                                calculate_width(self, screen_width, i)
                                - calculate_offset(state["hand"], j, tile_size)
                            ),
                            calculate_height(screen_height, 4, 1, tile_size, -1),
                        ),
                    )
                # Players with odd id go below public cards
                else:
                    self.screen.blit(
                        card_img,
                        (
                            (
                                calculate_width(self, screen_width, i)
                                - calculate_offset(state["hand"], j, tile_size)
                            ),
                            calculate_height(screen_height, 4, 3, tile_size, 0),
                        ),
                    )

            # Load and blit text for player name
            font = get_font(os.path.join("font", "Minecraft.ttf"), 36)
            text = font.render("Player " + str(i + 1), True, white)
            textRect = text.get_rect()
            if i % 2 == 0:
                textRect.center = (
                    (
                        screen_width
                        / (np.ceil(len(self.possible_agents) / 2) + 1)
                        * np.ceil((i + 1) / 2)
                    ),
                    calculate_height(screen_height, 4, 1, tile_size, -(22 / 20)),
                )
            else:
                textRect.center = (
                    (
                        screen_width
                        / (np.ceil(len(self.possible_agents) / 2) + 1)
                        * np.ceil((i + 1) / 2)
                    ),
                    calculate_height(screen_height, 4, 3, tile_size, (23 / 20)),
                )
            self.screen.blit(text, textRect)

            # Load and blit number of poker chips for each player
            font = get_font(os.path.join("font", "Minecraft.ttf"), 24)
            text = font.render(str(state["my_chips"]), True, white)
            textRect = text.get_rect()

            # Calculate number of each chip
            total = state["my_chips"]
            height = 0
            for key in chips:
                num = total / chips[key]["value"]
                chips[key]["number"] = int(num)
                total %= chips[key]["value"]

                chip_img = get_image(os.path.join("img", chips[key]["img"]))
                chip_img = pygame.transform.scale(
                    chip_img, (int(tile_size / 2), int(tile_size * 16 / 45))
                )

                # Blit poker chip img
                for j in range(0, int(chips[key]["number"])):
                    if i % 2 == 0:
                        self.screen.blit(
                            chip_img,
                            (
                                (
                                    calculate_width(self, screen_width, i)
                                    + tile_size * (8 / 10)
                                ),
                                calculate_height(screen_height, 4, 1, tile_size, -1 / 2)
                                - ((j + height) * tile_size / 15),
                            ),
                        )
                    else:
                        self.screen.blit(
                            chip_img,
                            (
                                (
                                    calculate_width(self, screen_width, i)
                                    + tile_size * (8 / 10)
                                ),
                                calculate_height(screen_height, 4, 3, tile_size, 1 / 2)
                                - ((j + height) * tile_size / 15),
                            ),
                        )
                height += chips[key]["number"]

            # Blit text number
            if i % 2 == 0:
                textRect.center = (
                    (calculate_width(self, screen_width, i) + tile_size * (21 / 20)),
                    calculate_height(screen_height, 4, 1, tile_size, -1 / 2)
                    - ((height + 1) * tile_size / 15),
                )
            else:
                textRect.center = (
                    (calculate_width(self, screen_width, i) + tile_size * (21 / 20)),
                    calculate_height(screen_height, 4, 3, tile_size, 1 / 2)
                    - ((height + 1) * tile_size / 15),
                )
            self.screen.blit(text, textRect)

        # Load and blit public cards
        for i, card in enumerate(state["public_cards"]):
            card_img = get_image(os.path.join("img", card + ".png"))
            card_img = pygame.transform.scale(
                card_img, (int(tile_size * (142 / 197)), int(tile_size))
            )
            if len(state["public_cards"]) <= 3:
                self.screen.blit(
                    card_img,
                    (
                        (
                            (
                                ((screen_width / 2) + (tile_size * 31 / 616))
                                - calculate_offset(state["public_cards"], i, tile_size)
                            ),
                            calculate_height(screen_height, 2, 1, tile_size, -(1 / 2)),
                        )
                    ),
                )
            else:
                if i <= 2:
                    self.screen.blit(
                        card_img,
                        (
                            (
                                (
                                    ((screen_width / 2) + (tile_size * 31 / 616))
                                    - calculate_offset(
                                        state["public_cards"][:3], i, tile_size
                                    )
                                ),
                                calculate_height(
                                    screen_height, 2, 1, tile_size, -21 / 20
                                ),
                            )
                        ),
                    )
                else:
                    self.screen.blit(
                        card_img,
                        (
                            (
                                (
                                    ((screen_width / 2) + (tile_size * 31 / 616))
                                    - calculate_offset(
                                        state["public_cards"][3:], i - 3, tile_size
                                    )
                                ),
                                calculate_height(
                                    screen_height, 2, 1, tile_size, 1 / 20
                                ),
                            )
                        ),
                    )

        if self.render_mode == "human":
            pygame.display.update()

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

   
class meta_wrapper(BaseWrapper):
    """
    A wrapper class for the environment that converts the multiagent environment into a single-player environment.

    This wrapper extends the functionality of a base RL environment by providing additional features related to
    observations, actions, and meta-information for two players in a game.
    
     It also calculates and tracks optimal actions
    based on hand evaluation scores.

    Args:
        env: The base RL environment to be wrapped.
        learner: The player for which the reinforcement learning agent is acting.
        obs_type (str): The observation type to use.

    Attributes:
        learner: The player for which the reinforcement learning agent is acting.
        baked (str): The player ID for the opponent
        players: The list of players in the game.
        game: The game object representing the environment's game.
        game_pointer_pu: The game pointer for public updates.
        obs_type (str): The selected observation type.
        AGENT: The learner's player object.
        OPPONENT: The opponent player object.
        observation_space: The observation space.
        action_space: The action space.
        opt_acts_ag (list): List to store optimal actions taken by the agent.
        opt_acts_op (list): List to store optimal actions taken by the opponent.

    Methods:
        observe(): Observe the current state of the environment.
        optimal_action(action, player): Calculate and track optimal actions based on hand evaluation scores.
        step(action): Perform one step in the environment, including optimal action calculation.
        reset(seed=None): Reset the environment and return the initial observation and info.
    """
    def __init__(self, env, learner, obs_type):
        super().__init__(env)
        self.learner = learner
        self.baked = 'player_0'
        self.players = env.env.env.env.env.game.players
        self.game = env.env.env.env.env.game
        self.game_pointer_pu = env.env.env.env.env.game.game_pointer
        self.obs_type = obs_type
        
        self.AGENT = self.players[0]
        self.OPPONENT = self.players[1]
        self.add_env_to_agents(env)
        self.observation_space = super().observation_space(self.learner)
        self.action_space = super().action_space(self.learner)
        self.opt_acts_ag = []
        self.opt_acts_op = []

    def observe(self):
        """
        Observe the current state of the environment.

        Returns:
            list: A list containing observations, cumulative rewards, terminations, truncations, and information.
        """
        return [super().observe(self.learner), self._cumulative_rewards[self.learner],
                self.terminations[self.learner], self.truncations[self.learner], self.infos[self.learner]]

    def optimal_action(self, action, player):
        """
        Calculate and track optimal actions based on hand evaluation scores.

        Args:
            action: The action taken by the player.
            player: The player for whom the optimal action is calculated.

        Returns:
            None
        """
        score_max = 7462
        quartiles = [score_max * 0.25, score_max * 0.5, score_max * 0.75]
        game = self.game

        # format cards in players hand
        hand = []
        for c in player.hand:
            c1r = c.rank
            c1s = c.suit.lower()
            c1 = c1r + c1s
            hand.append(c1)
        # format cards in community cards
        pc = []
        if len(game.public_cards) > 0:
            public_cards = game.public_cards

            for c in public_cards:
                cr_temp = c.rank
                cs_temp = c.suit.lower()
                pc.append(cr_temp + cs_temp)

        hand_objs = []
        pc_objs = []
        for c in hand:
            hand_objs.append(Card.new(c))
        for c in pc:
            pc_objs.append(Card.new(c))

        # if more than three cards in the hand (afer pre-flop round), deterime optimal action. If its the same as the action
        # taken by the agent or oppoent, append the op_acts list with a 1 to record this, or append with a zero if optimal action 
        # was not chosen. 
        if len(pc) >= 3:
            evaluator = Evaluator()
            try:
                score = evaluator.evaluate(hand_objs, pc_objs)
            except KeyError:
                score = 0

            if score <= quartiles[0]:
                op_act = 1
            if score >= quartiles[0] and score <= quartiles[1]:
                op_act = 0
            if score >= quartiles[1] and score <= quartiles[2]:
                op_act = 3
            if score >= quartiles[2]:
                op_act = 2
            if action == op_act:
                if player.player_id == 'player_1':
                    self.opt_acts_ag.append(1)
                if player.player_id == 'player_0':
                    self.opt_acts_op.append(1)
            else:
                if player.player_id == 'player_1':
                    self.opt_acts_ag.append(0)
                if player.player_id == 'player_0':
                    self.opt_acts_op.append(0)

    def step(self, action):
        """
        Perform one step in the environment by the agent. if the next observation is not terminated or truncated, 
        pass observation to the opponent and step the environment. append opponent rewardz (purposefully misspelt to avoid clashing with class
        attributes). 
        including optimal action calculation.

        Args:
            action: The action taken by the learner.

        Returns:
            list: A list containing observations, cumulative rewards, terminations, truncations, and information to the agent.
        """
        self.optimal_action(action, self.AGENT)
        super().step(action)
        if self.agent_selection != self.learner and not self.observe()[2] and not self.observe()[3]:
            op_action_mask = self.observe()[0]['action_mask']
            op_obs = super().observe(self.baked)
            ops_action = self.OPPONENT.get_action(op_obs, self.game)
            self.optimal_action(action, self.OPPONENT)
            super().step(ops_action)
            op_reward = self._cumulative_rewards[self.baked]
            self.OPPONENT.rewardz.append(op_reward)

        return self.observe()

    def reset(self, seed=None):
        """
        Reset the environment and return the initial observation and info.

        Args:
            seed (int, optional): A random seed for environment reset.

        Returns:
            tuple: A tuple containing the initial observation and an info dictionary.
        """
        super().reset(seed=seed)
        self.OPPONENT.rewardz = []
        obs, reward, done, truncation, info = self.observe()
        info = {'reward': reward, 'done': done, 'truncation': truncation, 'info': info}
        return (obs, info)
        
    def add_env_to_agents(self, env):
        for p in self.players:
            p.env = env
def env(obs_type, render_mode):
    """
    Create and configure an environment for reinforcement learning.

    This function sets up an environment applying a series
    of wrappers to the base environment. These wrappers modify the behavior of the
    environment to enforce certain rules or constraints.

    Parameters:
    - obs_type (str): The type of observation for the environment. This can be one of the
      supported observation types.
    - render_mode (str): The rendering mode for the environment. This specifies how the
      environment should be visually rendered, if at all.

    Returns:
    - env: A configured reinforcement learning environment with the specified observation
      type and rendering mode, and additional wrappers for rule enforcement.
    """
 
    env = raw_env(num_players=2, render_mode= render_mode, obs_type = obs_type)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    env = meta_wrapper(env, learner = 'player_1', obs_type = obs_type)
    return env

        