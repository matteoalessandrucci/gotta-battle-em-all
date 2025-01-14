from copy import deepcopy
import math
from typing import List
from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import GameState, PkmTeam

def n_fainted(t: PkmTeam):
    fainted = 0
    fainted += t.active.hp == 0
    if len(t.party) > 0:
        fainted += t.party[0].hp == 0
    if len(t.party) > 1:
        fainted += t.party[1].hp == 0
    return fainted


def game_state_eval(s: GameState, depth):
    mine = s.teams[0].active
    opp = s.teams[1].active
    return mine.hp / mine.max_hp - 3 * opp.hp / opp.max_hp - 0.3 * depth

class MinimaxNode:
    """
    Represents a node in the Minimax game tree.
    """
    def __init__(self, state, action=None, parent=None, depth=0):
        self.state = state         # Game state at this node
        self.action = action       # Action taken to reach this state
        self.parent = parent       # Parent node
        self.depth = depth         # Depth of the node in the tree
        self.children = []         # Child nodes
        self.eval_value = None     # Evaluation score of this node

class Minimax_Agent(BattlePolicy):
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def minimax(self, env, enemy_action, depth, alpha, beta, maximizing_player):
            if depth == 0:
                return game_state_eval(env, depth), None

            best_action = None
            if maximizing_player:
                max_eval = -math.inf
                for action in range(1, DEFAULT_N_ACTIONS):
                    g_copy = deepcopy(env)
                    g_copy.step([action, enemy_action])
                    eval_value, _ = self.minimax(g_copy, action, depth - 1, alpha, beta, False)
                    if eval_value > max_eval:
                        max_eval = eval_value
                        best_action = action
                    alpha = max(alpha, eval_value)
                    if beta <= alpha:
                        break
                return max_eval, best_action
            else:
                min_eval = math.inf
                for action in range(DEFAULT_N_ACTIONS):
                    g_copy = deepcopy(env)
                    g_copy.step([enemy_action, action])
                    eval_value, _ = self.minimax(g_copy, action, depth - 1, alpha, beta, True)
                    if eval_value < min_eval:
                        min_eval = eval_value
                        best_action = action
                    beta = min(beta, eval_value)
                    if beta <= alpha:
                        break
                return min_eval, best_action

    def get_action(self, g) -> int:
        _, action = self.minimax(g, 0, self.max_depth, -math.inf, math.inf, True)
        return action

class MinimaxNodes_Agent(BattlePolicy):
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth

    def minimax(self, node, enemy_action, depth, alpha, beta, maximizing_player):
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
        if depth == 0 :#or state.is_terminal():
            node.eval_value = game_state_eval(state, depth)
            return node.eval_value, node.action

        best_action = None

        if maximizing_player:
            max_eval = -math.inf
            for action in range(1, DEFAULT_N_ACTIONS):
                child_state = deepcopy(state)
                child_state.step([action, enemy_action])

                # Create a child node for this action
                child_node = MinimaxNode(child_state, action=action, parent=node, depth=node.depth + 1)
                node.children.append(child_node)

                eval_value, _ = self.minimax(child_node, action, depth - 1, alpha, beta, False)

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
                child_node = MinimaxNode(child_state, action=action, parent=node, depth=node.depth + 1)
                node.children.append(child_node)

                eval_value, _ = self.minimax(child_node, action, depth - 1, alpha, beta, True)

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
        _, best_action = self.minimax(root_node, 0, self.max_depth, -math.inf, math.inf, True)
        return best_action