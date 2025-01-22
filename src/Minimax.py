import math
import sys
import os
from utils import MinimaxNode, minimax

sys.path.append(os.path.join(sys.path[0], ".."))
from vgc.behaviour import BattlePolicy

class Minimax_agent(BattlePolicy):
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def get_action(self, game_state) -> int:
        """
        Determines the best action to take using Minimax with Alpha-Beta Pruning.
        :param game_state: Current state of the game.
        :return: The chosen action.
        """
        root_node = MinimaxNode(game_state)
        _, best_action = minimax(
            root_node, 0, self.max_depth, -math.inf, math.inf, True
        )
        return best_action