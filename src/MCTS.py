from typing import List
import numpy as np
from collections import namedtuple
from copy import deepcopy
import os
import sys

sys.path.append(os.path.join(sys.path[0], ".."))

from vgc.behaviour.BattlePolicies import BattlePolicy, BFSNode
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition
from vgc.engine.PkmBattleEnv import PkmTeam

class MCTSNode:
    def __init__(self, g=None, parent=None, action=None):
        self.g = g  # GameState
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.action = action  # Action that led to this node
        self.visits = 0  # Number of visits
        self.value = 0.0  # Cumulative reward


class MonteCarloAgent(BattlePolicy):
    """
    Monte Carlo Tree Search (MCTS) agent that uses simulations to decide on the best action to take.
    """

    def __init__(self, simulations: int = 30, exploration_constant: float = 1.4, max_turns: int = 15):
        """
        Initialize the MCTS Agent.

        :param simulations: Number of simulations per decision.
        :param exploration_constant: Trade-off factor for exploration vs. exploitation in UCB.
        :param max_turns: Maximum depth for simulations to prevent infinite loops.
        """
        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self.max_turns = max_turns

    def get_action(self, g) -> int:
        """
        Decide the best action to take from the current game state.
        :param g: Current game state (PkmBattleEnv).
        :return: Best action to take based on MCTS.
        """
        root = MCTSNode(g=g)

        # Perform simulations
        for _ in range(self.simulations):
            selected_node = self.select(root)  # Selection phase
            expanded_node = self.expand(selected_node)  # Expansion phase
            reward = self.simulate(expanded_node)  # Simulation phase
            self.backpropagate(expanded_node, reward)  # Backpropagation phase

        # Select the best action based on the highest visit count
        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.action

    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a child node based on UCB (Upper Confidence Bound) criteria, ensuring that all nodes
        are explored at least once.
        """
        # Prioritize unvisited nodes if there are any
        unvisited_nodes = [child for child in node.children if child.visits == 0]
        
        if unvisited_nodes:
            return unvisited_nodes[0]  # Select the first unvisited node
        
        if not node.children:  # If there are no children, expand the node
            return self.expand(node)  # Expanding will add children
        
        # If all nodes are visited, select using UCB
        selected_node = max(
            node.children,
            key=lambda child: (
                (child.value / child.visits if child.visits > 0 else 0)
                + self.exploration_constant
                * np.sqrt(np.log(node.visits + 1) / (child.visits + 1))
            ),
        )
        return selected_node


    def expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: Generate child nodes for unexplored actions, ensuring every action is explored.
        :param node: Node to expand.
        :return: Best or randomly selected child node for simulation.
        """
        if not node.children:  # Expand only if no children exist
            for action in range(6):
                child_g = deepcopy(node.g)
                opponent_action = self.simulate_opponent_action()
                child_g.step([action, opponent_action])  # Simulate the pair of actions
                child_node = MCTSNode(g=child_g, parent=node, action=action)
                node.children.append(child_node)

        # Ensure all children are evaluated before picking one
        return max(node.children, key=lambda n: self.evaluate_state(n.g))

    def simulate(self, node: MCTSNode) -> float:
        """
        Simulation phase: Run a random game simulation from the current state and evaluate the result.
        :param node: Node to start the simulation from.
        :return: Reward obtained from the simulation.
        """
        sim_g = deepcopy(node.g)
        t = False  # Terminal state flag
        turns = 0
        while not t and turns < self.max_turns:
            actions = self.select_simulation_actions(sim_g)
            _, _, t, _, _ = sim_g.step(actions)
            turns += 1
        return self.evaluate_state(sim_g, depth=turns)

    def backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: Update node values and visit counts along the path.
        :param node: Node where the reward is backpropagated.
        :param reward: Reward obtained from simulation.
        """
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def simulate_opponent_action(self) -> int:
        """
        Simulates an opponent's action using a more realistic heuristic (e.g., random weighted selection).
        :return: Chosen opponent action.
        """
        return np.random.choice(6)

    def select_simulation_actions(self, game_state: GameState) -> list:
        """
        Selects actions for the simulation phase. This can be adapted to be more strategic.
        :param game_state: Current state to evaluate.
        :return: List of actions for the simulation.
        """
        return [np.random.choice(6), np.random.choice(6)]  # Random opponent actions can be refined later

    # def evaluate_state(self, game_state: GameState, depth: int = 0) -> float:
    #     """
    #     Evaluate the current game state.
    #     :param game_state: Current game state to evaluate.
    #     :param depth: Depth of the simulation (penalty for deeper states).
    #     :return: Evaluation score of the game state.
    #     """
    #     ally = game_state.teams[0]
    #     opp = game_state.teams[1]

    #     # HP difference (with increased weight)
    #     score = ally.active.hp / ally.active.max_hp - 3 * opp.active.hp / opp.active.max_hp

    #     # Difference in fainted Pokémon (with increased weight)
    #     score += 15 * (3 - n_fainted(opp) - (3 - n_fainted(ally)))

    #     # Bonus for type effectiveness
    #     score += TYPE_CHART_MULTIPLIER[ally.active.type][opp.active.type] * 10

    #     # Penalty for low HP Pokémon
    #     if ally.active.hp / ally.active.max_hp < 0.2:
    #         score -= 10

    #     # Depth penalty
    #     score -= 0.05 * depth  # Reduced penalty per depth

    #     return score

    def evaluate_state(self, g: GameState, depth: int = 0) -> float:
        """Valuta lo stato di gioco per una determinata profondità."""
        my_active = g.teams[0].active
        opp_active = g.teams[1].active

        # Differenze di HP (valore normale fra 0 e 1 per ciascun Pokémon)
        hp_difference = my_active.hp / my_active.max_hp - opp_active.hp / opp_active.max_hp

        # Differenza nel numero di Pokémon svenuti per squadra
        fainted_difference = n_fainted(g.teams[1]) - n_fainted(g.teams[0])

        # Penalità per stati alterati (avvelenamento, paralisi, ecc.)
        my_status_penalty = 1 if my_active.status else 0
        opp_status_penalty = 1 if opp_active.status else 0

        # Calcolo totale del reward
        return (
            10 * fainted_difference +  # Favorire svenimenti avversari
            5 * hp_difference -  # Mantenere più HP rispetto all'avversario
            2 * my_status_penalty +  # Penalità se il Pokémon del giocatore ha status negativi
            2 * opp_status_penalty -  # Premia lo status negativo dell'avversario
            0.3 * depth  # Penalità per simulazioni profonde
        )


def n_fainted(team: PkmTeam) -> int:
    return sum(pkm.hp == 0 for pkm in [team.active] + team.party[:2])
