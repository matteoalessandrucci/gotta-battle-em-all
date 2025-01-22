from typing import List
import numpy as np
from vgc.behaviour.BattlePolicies import BattlePolicy, BFSNode
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition, PkmStatus
from collections import namedtuple
from vgc.engine.PkmBattleEnv import PkmTeam
from copy import deepcopy

import time


class RandomAgent(BattlePolicy):
    """
    Agent that selects actions randomly.
    """

    def __init__(self, switch_probability: float = .15, n_moves: int = 4,
                 n_switches: int = 2):
        super().__init__()
        self.n_actions: int = n_moves + n_switches
        self.n_switches = n_switches
        self.pi: List[float] = ([(1. - switch_probability) / n_moves] * n_moves) + (
                [switch_probability / n_switches] * n_switches)

    def get_action(self, g: GameState) -> int:
        my_type = g.teams[0].active.type
        opponent_type = g.teams[1].active.type
       # print(PkmType(my_type).name, PkmType(opponent_type).name)
        if(TYPE_CHART_MULTIPLIER[opponent_type][my_type] == 2):
            move = self.n_actions - self.n_switches + np.random.choice(self.n_switches)
          #  print("superefficacia: ", TYPE_CHART_MULTIPLIER[opponent_type][my_type], " mossa: ", move)
            return move 
        return np.random.choice(self.n_actions, p=self.pi)
    
class MinimaxWithAlphaBeta(BattlePolicy):
    """
    Tree search algorithm using Minimax with Alpha-Beta pruning for optimization in adversarial paradigms.
    """

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def minimax(self, node, depth, alpha, beta, maximizing_player) -> int:
        """ Helper function that implements Minimax with Alpha-Beta pruning. """
        # Condizioni di base del nodo (se ha raggiunto la profondità massima o il gioco è finito)
        # Verifica se il gioco è finito
        sim_g = deepcopy(node.g)
        t = False  # Flag terminale per il gioco
        max_turns = 10  # Limita la profondità della simulazione a 10 turni
        turns = 0
        while not t and turns < max_turns:
            actions = [np.random.choice(6) for _ in range(2)]
            _, _, t, _, _ = sim_g.step(actions)
            turns += 1
        
        # Valutazione dello stato finale del gioco
        if depth == self.max_depth or t:
            return game_state_eval(sim_g, depth)

        if maximizing_player:
            max_eval = float('-inf')
            # Espandi tutte le azioni possibili per questo giocatore
            for i in range(6):
                g = deepcopy(node.g)
                s, _, t, _, _ = g.step([i, 0])  # Solo la scelta dell'attacco dell'utente
                if n_fainted(s[0].teams[0]) > n_fainted(node.g.teams[0]):
                    continue
                eval = self.minimax(self._create_new_node(node, i, s[0], depth + 1), depth + 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Potatura
            return max_eval
        else:  # Minimizing player (l'avversario)
            min_eval = float('inf')
            for j in range(6):
                g = deepcopy(node.g)
                s, _, t, _, _ = g.step([0, j])  # Solo la scelta dell'avversario
                if n_fainted(s[0].teams[1]) > n_fainted(node.g.teams[1]):
                    continue
                eval = self.minimax(self._create_new_node(node, j, s[0], depth + 1), depth + 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Potatura
            return min_eval

    def _create_new_node(self, parent_node, action, game_state, depth):
        """ Crea e restituisci un nuovo nodo basato sull'azione effettuata e sullo stato di gioco risultante. """
        node = BFSNode()
        node.parent = parent_node
        node.a = action
        node.g = game_state
        node.depth = depth
        return node

    def get_action(self, g) -> int:
        root = BFSNode()
        root.g = g
        best_action = -1
        best_value = float('-inf')

        for i in range(6):
            g = deepcopy(root.g)
            s, _, t, _, _ = g.step([i, 0])  # Sposta il giocatore
            # Usa minimax con alpha-beta
            action_value = self.minimax(self._create_new_node(root, i, s[0], 1), 1, float('-inf'), float('inf'), False)
            if action_value > best_value:
                best_value = action_value
                best_action = i
        return best_action



class MonteCarloAgent(BattlePolicy):
    def __init__(self, simulations: int = 5, exploration_constant: float = 1.2):
        self.simulations = simulations
        self.exploration_constant = exploration_constant

    def get_action(self, g) -> int:
        root = MCTSNode(g=g)

        for _ in range(self.simulations):
            selected_node = self.select(root)  # Selection phase
            expanded_node = self.expand(selected_node)  # Expansion phase
            reward = self.simulate(expanded_node)  # Simulation phase
            self.backpropagate(expanded_node, reward)  # Backpropagation phase

        # Select the best action based on the highest visit count
        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.action

    def select(self, node: "MCTSNode") -> "MCTSNode":
        while node.children:
            node = max(
                node.children,
                key=lambda child: (
                    (child.value / child.visits if child.visits > 0 else float('inf')) +
                    self.exploration_constant * np.sqrt(np.log(node.visits + 1) / (child.visits + 1))
                )
            )
        return node

    def expand(self, node: "MCTSNode") -> "MCTSNode":
        if not node.children:  # Expand only if no children exist
            for action in range(6):
                child_g = deepcopy(node.g)
                opponent_action = np.random.choice(6)
                child_g.step([action, opponent_action])  # Simulate the pair of actions
                child_node = MCTSNode(g=child_g, parent=node, action=action)
                node.children.append(child_node)
        return np.random.choice(node.children)  # Choose a random child to simulate next

    def simulate(self, node: "MCTSNode") -> float:
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
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent


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


