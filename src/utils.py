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
        self.state = state  # Game state at this node
        self.action = action  # Action taken to reach this state
        self.parent = parent  # Parent node
        self.depth = depth  # Depth of the node in the tree
        self.children = []  # Child nodes
        self.eval_value = None  # Evaluation score of this node


class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search tree.
    """

    def __init__(self, g=None, parent=None, action=None):
        self.g = g  # GameState
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.action = action  # Action that led to this node
        self.visits = 0  # Number of visits
        self.value = 0.0  # Cumulative reward


def n_fainted(team: PkmTeam) -> int:
    return sum(pkm.hp == 0 for pkm in [team.active] + team.party[:2])


## evaluation function used in minimax
def evaluate_game_state_minimax(game_state: GameState) -> float:
    # Get the agent's team (player 0) and the opponent's team (player 1)
    agent = game_state.teams[0]
    opp = game_state.teams[1]

    # Initialize the score with the health advantage
    # Calculate the total HP of the agent's team (active + party) minus the opponent's
    score = sum(pkm.hp for pkm in agent.party + [agent.active]) - sum(
        pkm.hp for pkm in opp.party + [opp.active]
    )

    # Add a reward for having more Pokémon left in the party
    # Each extra Pokémon provides a significant advantage (weighted by 100)
    score += 100 * (len(agent.party) - len(opp.party))

    # Add a reward for knocking out the opponent's Pokémon
    # Rewards are based on the difference in the number of fainted Pokémon
    score += 50 * (
        len([pkm for pkm in opp.party if pkm.hp <= 0])  # Opponent's fainted Pokémon
        - len([pkm for pkm in agent.party if pkm.hp <= 0])  # Agent's fainted Pokémon
    )

    # Add a reward or penalty based on the type advantage of the active Pokémon
    # Uses a type effectiveness multiplier (e.g., Fire > Grass = 2.0, Water > Fire = 2.0)
    score += TYPE_CHART_MULTIPLIER[agent.active.type][opp.active.type] * 10

    # Return the final score; higher values indicate a more favorable state for the agent
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
        10 * fainted_difference  # High weigth for fainted pokemon
        + 5 * hp_difference  # we want high HP for us and low for opponent
        - 2 * my_status_penalty
        + 2 * opp_status_penalty
        - 0.3 * depth  # penalty for deeper nodes
    )


def minimax(node: MinimaxNode, enemy_action, depth, alpha, beta, maximizing_player):
    """
    Minimax algorithm with Alpha-Beta Pruning.
    :param node: Current MinimaxNode, representing the current state in the game tree.
    :param enemy_action: Action of the opponent in the current state.
    :param depth: Remaining depth to explore in the tree.
    :param alpha: Best score that the maximizing player is guaranteed to achieve.
    :param beta: Best score that the minimizing player is guaranteed to achieve.
    :param maximizing_player: Boolean indicating if it's the maximizing player's turn.
    :return: (evaluation score, best action) for the current node.
    """
    state = node.state

    # Terminal condition: If depth is 0 or the game is over, evaluate the current state.
    if depth == 0:
        node.eval_value = evaluate_game_state_minimax(state)  # Evaluate the state.
        return node.eval_value, node.action

    best_action = None  # Track the best action at this node.

    if maximizing_player:
        # Maximizing player's turn
        max_eval = -math.inf  # Start with the worst possible score for maximizing.
        for action in range(1, DEFAULT_N_ACTIONS):  # Explore all possible actions.
            # Simulate the result of the action.
            child_state = deepcopy(state)
            child_state.step([action, enemy_action])

            # Create a child node representing this action.
            child_node = MinimaxNode(
                child_state, action=action, parent=node, depth=node.depth
            )
            node.children.append(
                child_node
            )  # Add the child node to the current node's children.

            # Recursively call minimax for the child node (switch to minimizing player).
            eval_value, _ = minimax(child_node, action, depth - 1, alpha, beta, False)

            # Update the maximum evaluation value and track the best action.
            if eval_value > max_eval:
                max_eval = eval_value
                best_action = action

            # Update alpha (the best score for maximizing so far).
            alpha = max(alpha, eval_value)

            # Prune the branch if beta <= alpha (no need to explore further).
            if beta <= alpha:
                break  # Alpha-Beta Pruning

        # Store the best evaluation value in the current node and return.
        node.eval_value = max_eval
        return max_eval, best_action

    else:
        # Minimizing player's turn
        min_eval = math.inf  # Start with the worst possible score for minimizing.
        for action in range(DEFAULT_N_ACTIONS):  # Explore all possible actions.
            # Simulate the result of the action.
            child_state = deepcopy(state)
            child_state.step([enemy_action, action])

            # Create a child node representing this action.
            child_node = MinimaxNode(
                child_state, action=action, parent=node, depth=node.depth
            )
            node.children.append(
                child_node
            )  # Add the child node to the current node's children.

            # Recursively call minimax for the child node (switch to maximizing player).
            eval_value, _ = minimax(child_node, action, depth - 1, alpha, beta, True)

            # Update the minimum evaluation value and track the best action.
            if eval_value < min_eval:
                min_eval = eval_value
                best_action = action

            # Update beta (the best score for minimizing so far).
            beta = min(beta, eval_value)

            # Prune the branch if beta <= alpha (no need to explore further).
            if beta <= alpha:
                break  # Alpha-Beta Pruning

        # Store the best evaluation value in the current node and return.
        node.eval_value = min_eval
        return min_eval, best_action
