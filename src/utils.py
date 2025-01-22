from copy import deepcopy
import math
import sys
import os
import numpy as np
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
    def __init__(self, g=None, parent=None, action=None):
        self.g = g  # GameState
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.action = action  # Action that led to this node
        self.visits = 0  # Number of visits
        self.value = 0.0  # Cumulative reward
        self.depth = parent.depth + 1 if parent else 0  # Calculate depth based on parent


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

def adapt_parameters(self, g: GameState):
        """
        Adapt parameters dynamically based on game progress.
        """
        # Calculate progress based on remaining Pokémon
        my_remaining = sum(pkm.hp > 0 for pkm in g.teams[0].party + [g.teams[0].active])
        opp_remaining = sum(pkm.hp > 0 for pkm in g.teams[1].party + [g.teams[1].active])
        total_pokemon = len(g.teams[0].party) + len(g.teams[1].party) + 2
        progress = (total_pokemon - (my_remaining + opp_remaining)) / total_pokemon

        # Adapt simulations and exploration constant
        simulations = int(min_simulations + (self.max_simulations - self.min_simulations) * progress)
        self.exploration_constant = (
            self.max_exploration_constant
            - (self.max_exploration_constant - self.min_exploration_constant) * progress
        )
    
def adapt_parameters_inverse(self, g: GameState):
    """
    Adapt parameters dynamically based on game progress.
    """
    # Calculate progress based on remaining Pokémon
    my_remaining = sum(pkm.hp > 0 for pkm in g.teams[0].party + [g.teams[0].active])
    opp_remaining = sum(pkm.hp > 0 for pkm in g.teams[1].party + [g.teams[1].active])
    total_pokemon = len(g.teams[0].party) + len(g.teams[1].party) + 2
    progress = (total_pokemon - (my_remaining + opp_remaining)) / total_pokemon

    # Adjust parameters
    self.simulations = int(self.max_simulations - (self.max_simulations - self.min_simulations) * progress)
    self.exploration_constant = (
        self.min_exploration_constant
        + (self.max_exploration_constant - self.min_exploration_constant) * progress
    )

def select(self, node: "MCTSNode") -> "MCTSNode":
    """
    Selection phase: Navigate the tree to find the best node to expand.
    """
    while node.children and not is_terminal(node):
        node = max(
            node.children,
            key=lambda child: (
                (child.value / child.visits if child.visits > 0 else float('inf')) +
                self.exploration_constant * np.sqrt(np.log(node.visits + 1) / (child.visits + 1))
            )
        )
    return node

def expand(self, node: "MCTSNode") -> "MCTSNode":
    """
    Expansion phase: Generate child nodes for unexplored actions.
    """
    if not node.children and not is_terminal(node):
        for action in range(6):
            child_g = deepcopy(node.g)
            opponent_action = np.random.choice(6)#self.heuristic_action(child_g.teams[1]) #np.random.choice(6)   # Opponent uses heuristic
            child_g.step([action, opponent_action])
            child_node = MCTSNode(g=child_g, parent=node, action=action)
            node.children.append(child_node)
    
    # Use an evaluation function to select the best child
    #best_child = max(node.children,key=lambda child: game_state_eval(child.g, depth=child.depth))
    random_child = np.random.choice(node.children)
    return random_child #node


def simulate(self, node: "MCTSNode") -> float:
    """
    Simulation phase: Run a heuristic-guided simulation from the current state.
    """
    sim_g = deepcopy(node.g)
    t = False
    max_turns = 12
    turns = 0

    while not t and turns < max_turns:
        actions = [
            self.heuristic_action(sim_g.teams[0]),  # Action based on heuristic for agent
            #np.random.choice(6),
            #self.heuristic_action(sim_g.teams[1])  # Action based on heuristic for opponent
            np.random.choice(6)  # Random action for opponent
        ]
        _, _, t, _, _ = sim_g.step(actions)
        turns += 1

    return game_state_eval(sim_g, depth=turns)
    #return self._evaluate_gamestate(sim_g)

def simulateAgne(self, node: "MCTSNode") -> float:
    sim_g = deepcopy(node.g)
    t = False  # Terminal state flag
    max_turns = 12  # Limit the depth of simulation to 10 turns
    turns = 0
    while not t and turns < max_turns:
        actions = [np.random.choice(6) for _ in range(2)]
        _, _, t, _, _ = sim_g.step(actions)
        turns += 1
    return game_state_eval(sim_g, depth=turns)

def backpropagate(self, node: "MCTSNode", reward: float):
    """
    Backpropagation phase: Update node values and visit counts along the path.
    """
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent

def heuristic_action(self, team) -> int:
    """
    A heuristic to decide an action for the agent.
    """
    actions = range(6)
    return max(actions, key=lambda action: self.evaluate_action(team, action))

def evaluate_action(self, team, action) -> float:
    """
    Evaluate the desirability of an action based on the team's state.
    """
    simulated_team = deepcopy(team)
    opponent_type = team.active.type.value  # Ottieni il valore numerico del tipo avversario
    move_type = self.get_move_type(action).value  # Ottieni il valore numerico del tipo della mossa

    #print(f"Tipo numerico di mossa: {move_type}")
    # Calcola il moltiplicatore di efficacia
    advantage = TYPE_CHART_MULTIPLIER[move_type][opponent_type]

    # Simula il danno (esempio basato sull'azione e sull'efficacia)
    damage = action * 5 * advantage
    simulated_team.active.hp -= damage

    return simulated_team.active.hp / simulated_team.active.max_hp

def get_move_type(self, action: int):
    """
    Returns the type of the move corresponding to the given action.
    This should map action indices to move types.
    """
    # Mappa le azioni ai tipi di mosse
    move_types = [
        PkmType.NORMAL,
        PkmType.FIRE,
        PkmType.WATER,
        PkmType.GRASS,
        PkmType.ELECTRIC,
        PkmType.ICE,
    ]
    
    # Determina il tipo di mossa corrispondente
    move_type = move_types[action % len(move_types)]
    
    # Stampa il tipo di mossa selezionata e l'azione
    #print(f"Action: {action}, Move Type: {move_type}")
    
    return move_type




def is_terminal( node: "MCTSNode") -> bool:
    """
    Check if the node represents a terminal state.
    """
    return all(pkm.hp == 0 for pkm in node.g.teams[0].party + [node.g.teams[0].active]) or \
            all(pkm.hp == 0 for pkm in node.g.teams[1].party + [node.g.teams[1].active])

def _evaluate_gamestate(self, game_state: GameState) -> float:
    """
    Evaluate the desirability of a game state for the agent.
    """
    ally = game_state.teams[0]
    opp = game_state.teams[1]
    score = sum(pkm.hp for pkm in ally.party + [ally.active]) - sum(pkm.hp for pkm in opp.party + [opp.active])
    score += 100 * (len(ally.party) - len(opp.party))
    score += 50 * (len([pkm for pkm in opp.party if pkm.hp <= 0]) - len([pkm for pkm in ally.party if pkm.hp <= 0]))
    score += TYPE_CHART_MULTIPLIER[ally.active.type][opp.active.type] * 10
    return score

def game_state_eval(g: GameState, depth: int) -> float:
    my_active = g.teams[0].active
    opp_active = g.teams[1].active

    # Normalized hp difference
    hp_difference = my_active.hp / my_active.max_hp - opp_active.hp / opp_active.max_hp

    # Difference between fainted pokemons
    fainted_difference = n_fainted(g.teams[1]) - n_fainted(g.teams[0])

    # def status_penalty(status: PkmStatus) -> float:
    #     penalties = {
    #         PkmStatus.NONE: 0,        
    #         PkmStatus.PARALYZED: 1,  
    #         PkmStatus.POISONED: 2,   
    #         PkmStatus.CONFUSED: 1,   
    #         PkmStatus.SLEEP: 1.5,    
    #         PkmStatus.FROZEN: 2,     
    #         PkmStatus.BURNED: 2.5    
    #     }
    #     print(penalties.get(status, 0))
    #     return penalties.get(status, 0)

    # my_status_penalty = status_penalty(my_active.status)
    # opp_status_penalty = status_penalty(opp_active.status)

    return (
        10 * fainted_difference +  
        5 * hp_difference -        
        # 2 * my_status_penalty +    
        # 2 * opp_status_penalty -   
        0.3 * depth                
    )