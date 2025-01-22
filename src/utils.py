from copy import deepcopy
import math
import sys
import os
sys.path.append(os.path.join(sys.path[0], ".."))

from vgc.datatypes.Objects import GameState
from vgc.engine.PkmBattleEnv import PkmTeam


from vgc.datatypes.Constants import DEFAULT_N_ACTIONS, TYPE_CHART_MULTIPLIER

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

class MCTSNode:
    def __init__(self, g=None, parent=None, action=None):
        self.g = g  # GameState
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.action = action  # Action that led to this node
        self.visits = 0  # Number of visits
        self.value = 0.0  # Cumulative reward

def n_fainted(team: PkmTeam) -> int:
    return sum(pkm.hp == 0 for pkm in [team.active] + team.party[:2])


def evaluate_game_state_minimax(game_state: GameState) -> float:
        agent = game_state.teams[0]
        opp = game_state.teams[1]
        score = sum(pkm.hp for pkm in agent.party + [agent.active]) - sum(
            pkm.hp for pkm in opp.party + [opp.active]
        )
        score += 100 * (len(agent.party) - len(opp.party))
        score += 50 * (
            len([pkm for pkm in opp.party if pkm.hp <= 0])
            - len([pkm for pkm in agent.party if pkm.hp <= 0])
        )
        score += TYPE_CHART_MULTIPLIER[agent.active.type][opp.active.type] * 10
        return score

def evaluate_game_state_MCTS(g: GameState, depth: int = 0) -> float:
        """Valuta lo stato di gioco per una determinata profondità."""
        my_active = g.teams[0].active
        opp_active = g.teams[1].active

        # HP differnce between pokemon (valore normale fra 0 e 1 per ciascun Pokémon)
        hp_difference = my_active.hp / my_active.max_hp - opp_active.hp / opp_active.max_hp

        # difference in remaining pokemon for each team
        fainted_difference = n_fainted(g.teams[1]) - n_fainted(g.teams[0])

        # Pokemon status penalties
        my_status_penalty = 1 if my_active.status else 0
        opp_status_penalty = 1 if opp_active.status else 0

        # final score
        return (
            10 * fainted_difference +  # High weigth for fainted pokemon
            5 * hp_difference -  # we want high HP for us and low for opponent
            2 * my_status_penalty +  
            2 * opp_status_penalty -
            0.3 * depth  # penalty for deeper nodes
        )

def minimax(node:MinimaxNode, enemy_action, depth, alpha, beta, maximizing_player):
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
            node.eval_value = evaluate_game_state_minimax(state)
            return node.eval_value, node.action

        best_action = None

        if maximizing_player:
            max_eval = -math.inf
            for action in range(1, DEFAULT_N_ACTIONS):
                child_state = deepcopy(state)
                child_state.step([action, enemy_action])

                # Create a child node for this action
                child_node = MinimaxNode(
                    child_state, action=action, parent=node, depth=node.depth
                )
                node.children.append(child_node)

                eval_value, _ = minimax(
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
                    child_state, action=action, parent=node, depth=node.depth
                )
                node.children.append(child_node)

                eval_value, _ = minimax(
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
