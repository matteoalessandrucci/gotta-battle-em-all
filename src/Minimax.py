import math
import sys
import os
from utils import MinimaxNode, minimax

sys.path.append(os.path.join(sys.path[0], ".."))
from vgc.behaviour import BattlePolicy


class Minimax_agent(BattlePolicy):
    def __init__(self, max_depth: int = 3):
        """
        Initializes the Minimax agent with a specified maximum search depth.
        :param max_depth: The maximum depth of the game tree to explore (default is 4).
        """
        self.max_depth = max_depth

    def get_action(self, game_state) -> int:
        """
        Determines the best action to take using the Minimax algorithm with Alpha-Beta Pruning.
        :param game_state: The current state of the game.
        :return: The chosen action to take.
        """
        # Create the root node for the Minimax tree using the current game state.
        root_node = MinimaxNode(game_state)

        # Call the Minimax function to find the best action.
        # Parameters:
        # - root_node: The root of the game tree representing the current state.
        # - 0: The opponent's action (initially assumed to be 0, may change based on the game).
        # - self.max_depth: The maximum depth to explore in the game tree.
        # - -math.inf: Initial alpha value (worst-case score for the maximizing player).
        # - math.inf: Initial beta value (worst-case score for the minimizing player).
        # - True: Indicates that it's the maximizing player's turn.
        _, best_action = minimax(
            root_node, 0, self.max_depth, -math.inf, math.inf, True
        )

        # Return the best action determined by the Minimax algorithm.
        return best_action
