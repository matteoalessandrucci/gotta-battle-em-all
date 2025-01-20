from typing import List
import copy
from copy import deepcopy
import numpy as np
import os
import sys
sys.path.append(os.path.join(sys.path[0], '..'))

from vgc.behaviour.BattlePolicies import BattlePolicy
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Types import PkmType
from vgc.engine.PkmBattleEnv import PkmTeam, PkmBattleEnv

import math
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER

class MinimaxNode:
    def __init__(self, state: GameState, action: int = None, parent=None, depth: int = 0):
        self.state = state
        self.action = action
        self.parent = parent
        self.depth = depth
        self.eval_value = None
        self.children = []

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


def game_state_eval(game_state: GameState, depth: int) -> float:
    """Valuta lo stato attuale del gioco."""
    ally = game_state.teams[0]
    opp = game_state.teams[1]
    
    # Differenza negli HP (con peso aumentato)
    score = ally.active.hp / ally.active.max_hp - 3 * opp.active.hp / opp.active.max_hp
    
    # Penalità per stato del Pokémon attivo
    # if ally.active.status != "PkmStatus.NONE":
    #     score -= 5
    # if opp.active.status = "PkmStatus.NONE":
    #     score += 10

    # Differenza nei Pokémon fainted (con peso aumentato)
    score += 15 * (3 - n_fainted(opp) - (3 - n_fainted(ally)))
    
    # Bonus per efficacia del tipo
    score += TYPE_CHART_MULTIPLIER[ally.active.type][opp.active.type] * 10

    # Penalità per Pokémon quasi KO
    if ally.active.hp / ally.active.max_hp < 0.2:
        score -= 10
    
    # Penalità per profondità
    return score - 0.1 * depth

# def game_state_eval(s: GameState, depth):
#     mine = s.teams[0].active
#     opp = s.teams[1].active
#     return mine.hp / mine.max_hp - 3 * opp.hp / opp.max_hp - 0.3 * depth

#cambia mcts_depth =2  e simulations=700
class MCTSxMinimax(BattlePolicy):
    #print("Agent class")
    def __init__(self, switch_probability: float = 0.15, n_moves: int = 4, n_switches: int = 2, max_depth: int = 2, simulations: int = 30, exploration_constant: float = 1.2, mcts_depth: int = 1):
        super().__init__()
        self.n_moves = n_moves
        self.n_switches = n_switches
        self.n_actions = n_moves + n_switches
        self.max_depth = max_depth
        self.pi = self._calculate_action_probabilities(switch_probability)
        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self.mcts_depth = mcts_depth  # Depth for MCTS simulation

    def _calculate_action_probabilities(self, switch_probability: float) -> List[float]:
        attack_prob = (1. - switch_probability) / self.n_moves
        switch_prob = switch_probability / self.n_switches
        return [attack_prob] * self.n_moves + [switch_prob] * self.n_switches

    def get_action(self, game_state: GameState) -> int:
        # Esegui MCTS
        mcts_root = self.run_mcts(game_state, self.mcts_depth)
        
        # Ordina i nodi figli del root in base al punteggio MCTS
        best_mcts_actions = sorted(
            mcts_root.children,
            key=lambda n: n.value / n.visits if n.visits > 0 else float('-inf'),
            reverse=True
        )

        best_action = None
        best_score = -math.inf

        for node in best_mcts_actions:
            # Valutazione tramite Minimax
            root_node = MinimaxNode(node.g)  # Nodo iniziale per il minimax
            eval_value, _ = self.minimax(root_node, 0, self.max_depth, -math.inf, math.inf, True)
            
            # Scegli l'azione con il miglior punteggio Minimax
            if eval_value > best_score:
                best_score = eval_value
                best_action = node.action


        return best_action



    def minimax(self, node: MinimaxNode, enemy_action: int, depth: int, alpha: float, beta: float, maximizing_player: bool):
        state = node.state
        if depth == 0 or self._is_terminal(state):
            node.eval_value = self._evaluate_gamestate(state)
            return node.eval_value, node.action

        best_action = None

        if maximizing_player:
            max_eval = -math.inf
            for action in range(self.n_actions):
                child_state = copy.deepcopy(state)
                child_state.step([action, enemy_action])
                child_node = MinimaxNode(child_state, action=action, parent=node, depth=node.depth + 1)
                node.children.append(child_node)

                eval_value, _ = self.minimax(child_node, action, depth - 1, alpha, beta, False)
                if eval_value > max_eval:
                    max_eval = eval_value
                    best_action = action

                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break
            node.eval_value = max_eval
            return max_eval, best_action
        else:
            min_eval = math.inf
            for action in range(self.n_actions):
                child_state = copy.deepcopy(state)
                child_state.step([action, enemy_action])
                child_node = MinimaxNode(child_state, action=action, parent=node, depth=node.depth + 1)
                node.children.append(child_node)

                eval_value, _ = self.minimax(child_node, action, depth - 1, alpha, beta, True)
                if eval_value < min_eval:
                    min_eval = eval_value
                    best_action = action

                beta = min(beta, eval_value)
                if beta <= alpha:
                    break
            node.eval_value = min_eval
            return min_eval, best_action


    def _evaluate_gamestate(self, game_state: GameState) -> float:
        ally = game_state.teams[0]
        opp = game_state.teams[1]
        score = sum(pkm.hp for pkm in ally.party + [ally.active]) - sum(pkm.hp for pkm in opp.party + [opp.active])
        score += 100 * (len(ally.party) - len(opp.party))
        score += 50 * (len([pkm for pkm in opp.party if pkm.hp <= 0]) - len([pkm for pkm in ally.party if pkm.hp <= 0]))
        score += TYPE_CHART_MULTIPLIER[ally.active.type][opp.active.type] * 10
        return score

    def _is_terminal(self, game_state: GameState) -> bool:
        return all(pkm.hp <= 0 for pkm in game_state.teams[0].party + [game_state.teams[0].active]) or \
               all(pkm.hp <= 0 for pkm in game_state.teams[1].party + [game_state.teams[1].active])

    def run_mcts(self, game_state: GameState, depth: int):
        root = MCTSNode(g=game_state)
        for i in range(self.simulations):
            selected_node = self.select(root)
            expanded_node = self.expand(selected_node, depth)
            reward = self.simulate(expanded_node)
            self.backpropagate(expanded_node, reward)
        return root

    def select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            node = max(
                node.children,
                key=lambda child: (
                    (child.value / child.visits if child.visits > 0 else float('inf')) +
                    self.exploration_constant * np.sqrt(np.log(node.visits + 1) / (child.visits + 1))
                )
            )
        return node


    def expand(self, node: MCTSNode, depth: int) -> MCTSNode:
        if not node.children or node.g.turn < depth:  
            for action in range(self.n_actions):
                child_g = deepcopy(node.g)
                child_g.step([action, np.random.choice(self.n_actions)])  # Simula con un'azione random dell'avversario
                child_node = MCTSNode(g=child_g, parent=node, action=action)
                node.children.append(child_node)

        if node.children:
            return np.random.choice(node.children)
        else:
            return node  # Ritorna il nodo corrente se non ci sono figli

    def simulate(self, node: MCTSNode) -> float:
        sim_g = deepcopy(node.g)
        t = False
        max_turns = 10
        turns = 0
        while not t and turns < max_turns:
            actions = [np.random.choice(self.n_actions) for _ in range(2)]
            _, _, t, _, _ = sim_g.step(actions)
            turns += 1
        return game_state_eval(sim_g, depth=turns)

    def backpropagate(self, node: MCTSNode, reward: float):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent