from typing import List
import numpy as np
from collections import namedtuple
from copy import deepcopy
import os
import sys
from utils import MCTSNode, game_state_eval

sys.path.append(os.path.join(sys.path[0], ".."))

from vgc.behaviour.BattlePolicies import BattlePolicy


class MonteCarloAgent(BattlePolicy):
    """
    Monte Carlo Tree Search (MCTS) agent that uses simulations to decide on the best action to take.
    """

    def __init__(self, simulations: int = 100, exploration_constant: float = 1.41):
        """
        Initialize the MCTS Agent.

        :param simulations: Number of simulations per decision.
        :param exploration_constant: Trade-off factor for exploration vs. exploitation in UCB.
        """
        self.simulations = simulations
        self.exploration_constant = exploration_constant

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

    def select(
        self, node: "MCTSNode"
    ) -> "MCTSNode":  # Naviga l'albero fino a trovare un nodo senza figli
        """
        Selection phase: Navigate the tree to find the best node to expand.
        :param node: Current node to start selection from.
        :return: Selected node for expansion.
        """
        while node.children:  # Traverse the tree while child nodes exist
            node = max(
                node.children,
                key=lambda child: (
                    child.value / child.visits
                    if child.visits > 0
                    else float("inf")
                    + self.exploration_constant
                    * np.sqrt(np.log(node.visits + 1) / (child.visits + 1))
                ),
            )
        return node

    def expand(self, node: "MCTSNode") -> "MCTSNode":
        """
        Expansion phase: Generate child nodes for unexplored actions.
        :param node: Node to expand.
        :return: Randomly selected child node for simulation.
        """
        if not node.children:  # Expand only if no children exist
            for action in range(6):
                child_g = deepcopy(node.g)
                opponent_action = np.random.choice(6)
                child_g.step([action, opponent_action])  # Simulate the pair of actions
                child_node = MCTSNode(g=child_g, parent=node, action=action)
                node.children.append(child_node)
        return np.random.choice(node.children)  # Choose a random child to simulate next

    def simulate(self, node: "MCTSNode") -> float:
        """
        Simulation phase: Run a random game simulation from the current state and evaluate the result.
        :param node: Node to start the simulation from.
        :return: Reward obtained from the simulation.
        """
        sim_g = deepcopy(node.g)
        t = False  # Terminal state flag
        max_turns = 10  # Limit the depth of simulation to 10 turns
        turns = 0
        while not t and turns < max_turns:
            actions = [np.random.choice(6) for _ in range(2)]
            _, _, t, _, _ = sim_g.step(actions)
            turns += 1
        return game_state_eval(sim_g, depth=turns)

    def backpropagate(self, node: "MCTSNode", reward: float):
        """
        Backpropagation phase: Update node values and visit counts along the path.
        :param node: Node where the reward is backpropagated.
        :param reward: Reward obtained from simulation.
        """
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
