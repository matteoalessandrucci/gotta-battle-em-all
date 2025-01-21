from copy import deepcopy
import math
from typing import List
import sys
import os
from utils import MinimaxNode, evaluate_game_state
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER

sys.path.append(os.path.join(sys.path[0], ".."))
from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import GameState, PkmTeam





class MinimaxNode:
    """
    Represents a node in the Minimax game tree.
    """

    def __init__(self, state, action=None, parent=None, depth=0):
        self.state = state  # Game state at this node
        self.action = action  # Action taken to reach this state
        self.parent = parent  # Parent node
        self.depth = depth  # Depth of the node in the tree
        self.children = []  # Child nodes
        self.eval_value = None  # Evaluation score of this node


class MinimaxNodes_Agent(BattlePolicy):
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth

    def minimax(self, node:MinimaxNode, enemy_action, depth, alpha, beta, maximizing_player):
        """
        Minimax algorithm with Alpha-Beta Pruning.
        :param node: Current MinimaxNode.
        :param enemy_action: Action of the opponent.
        :param depth: Current depth in the tree.
        :param alpha: Best already explored option along the path to the root for the maximizing player.
        :param beta: Best already explored option along the path to the root for the minimizing player.
        :param maximizing_player: Boolean indicating whether it's the maximizing player's turn.
        :return: (evaluation score, best action)
        """
        state = node.state

        # Terminal condition: depth limit or end game
        if depth == 0:  # or state.is_terminal():
            node.eval_value = self.game_state_eval(state)
            return node.eval_value, node.action

        best_action = None

        if maximizing_player:
            max_eval = -math.inf
            for action in range(1, DEFAULT_N_ACTIONS):
                child_state = deepcopy(state)
                child_state.step([action, enemy_action])

                # Create a child node for this action
                child_node = MinimaxNode(
                    child_state, action=action, parent=node, depth=node.depth + 1
                )
                node.children.append(child_node)

                eval_value, _ = self.minimax(
                    child_node, action, depth - 1, alpha, beta, False
                )

                if eval_value > max_eval:
                    max_eval = eval_value
                    best_action = action

                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break  # Alpha-Beta Pruning
            node.eval_value = max_eval
            return max_eval, best_action
        else:
            min_eval = math.inf
            for action in range(DEFAULT_N_ACTIONS):
                child_state = deepcopy(state)
                child_state.step([enemy_action, action])

                # Create a child node for this action
                child_node = MinimaxNode(
                    child_state, action=action, parent=node, depth=node.depth + 1
                )
                node.children.append(child_node)

                eval_value, _ = self.minimax(
                    child_node, action, depth - 1, alpha, beta, True
                )

                if eval_value < min_eval:
                    min_eval = eval_value
                    best_action = action

                beta = min(beta, eval_value)
                if beta <= alpha:
                    break  # Alpha-Beta Pruning
            node.eval_value = min_eval
            return min_eval, best_action

    def get_action(self, game_state) -> int:
        """
        Determines the best action to take using Minimax with Alpha-Beta Pruning.
        :param game_state: Current state of the game.
        :return: The chosen action.
        """
        root_node = MinimaxNode(game_state)
        _, best_action = self.minimax(
            root_node, 0, self.max_depth, -math.inf, math.inf, True
        )
        return best_action
    
    def game_state_eval(self, game_state: GameState) -> float:
        ally = game_state.teams[0]
        opp = game_state.teams[1]
        score = sum(pkm.hp for pkm in ally.party + [ally.active]) - sum(
            pkm.hp for pkm in opp.party + [opp.active]
        )
        score += 100 * (len(ally.party) - len(opp.party))
        score += 50 * (
            len([pkm for pkm in opp.party if pkm.hp <= 0])
            - len([pkm for pkm in ally.party if pkm.hp <= 0])
        )
        score += TYPE_CHART_MULTIPLIER[ally.active.type][opp.active.type] * 10
        return score
    