from typing import List
import numpy as np
from vgc.behaviour.BattlePolicies import BattlePolicy
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition
from collections import namedtuple
from vgc.engine.PkmBattleEnv import PkmTeam
from copy import deepcopy

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
        print(PkmType(my_type).name, PkmType(opponent_type).name)
        if(TYPE_CHART_MULTIPLIER[opponent_type][my_type] == 2):
            move = self.n_actions - self.n_switches + np.random.choice(self.n_switches)
            print("superefficacia: ", TYPE_CHART_MULTIPLIER[opponent_type][my_type], " mossa: ", move)
            return move 
        return np.random.choice(self.n_actions, p=self.pi)

# class MonteCarloAgent(BattlePolicy):
#     """
#     Monte Carlo Tree Search (MCTS) agent that uses simulations to decide on the best action to take.
#     """

#     def __init__(self, simulations: int = 100, exploration_constant: float = 1.41):
#         self.simulations = simulations  # Number of simulations per action
#         self.exploration_constant = exploration_constant  # Exploration vs exploitation tradeoff constant

#     def get_action(self, g) -> int:  # g: PkmBattleEnv
#         root: MCTSNode = MCTSNode(g=g)

#         # Perform simulations
#         for _ in range(self.simulations):
#             selected_node = self.select(root)  # Selection phase
#             expanded_node = self.expand(selected_node)  # Expansion phase
#             reward = self.simulate(expanded_node)  # Simulation phase
#             self.backpropagate(expanded_node, reward)  # Backpropagation phase

#         # Select the best action based on the average reward
#         best_child = max(root.children, key=lambda n: n.value / n.visits if n.visits > 0 else 0)
#         print(best_child.action)
#         return best_child.action

#     def select(self, node: "MCTSNode") -> "MCTSNode":
#         """
#         Selection phase: Traverse the tree to find the most promising node using UCB1 formula.
#         """
#         while node.children:  # Keep selecting until a leaf node is reached
#             node = max(
#                 node.children,
#                 key=lambda child: (
#                     child.value / child.visits if child.visits > 0 else float('inf') +
#                     self.exploration_constant * (np.sqrt(np.log(node.visits) / (child.visits + 1)))
#                 )
#             )
#         return node

#     def expand(self, node: "MCTSNode") -> "MCTSNode":
#         """
#         Expansion phase: Expand the selected node by creating child nodes for all possible actions.
#         """
#         if not node.children:  # Avoid expanding a terminal node
#             for action in range(6):
#                 child_g = deepcopy(node.g)
#                 opponent_action = np.random.choice(6)
#                 child_g.step([action, opponent_action])  # Simulate the action pair
#                 child_node = MCTSNode(g=child_g, parent=node, action=action)
#                 node.children.append(child_node)
#         return np.random.choice(node.children)  # Select a random child to simulate next

#     def simulate(self, node: "MCTSNode") -> float:
#         """
#         Simulation phase: Simulate a random game from the current state and return the reward.
#         """
#         sim_g = deepcopy(node.g)
#         t = False  # Terminal state flag
#         while not t:  # Play until terminal state is reached
#             actions = [np.random.choice(6) for _ in range(2)]
#             _,_,terminal,_,_ = sim_g.step(actions)
#             t = terminal
#         return -n_fainted(sim_g.teams[1]) + n_fainted(sim_g.teams[0])

#     def backpropagate(self, node: "MCTSNode", reward: float):
#         """
#         Backpropagation phase: Update the value and visit count for all nodes in the path.
#         """
#         while node:
#             node.visits += 1
#             node.value += reward
#             node = node.parent

# # Utility class to represent MCTS nodes
# class MCTSNode:
#     def __init__(self, g=None, parent=None, action=None):
#         self.g = g  # GameState
#         self.parent = parent  # Parent node
#         self.children = []  # Child nodes
#         self.action = action  # Action taken to reach this state
#         self.visits = 0  # Number of times this node has been visited
#         self.value = 0.0  # Total value of this node

# def n_fainted(t: PkmTeam):
#     fainted = 0
#     fainted += t.active.hp == 0
#     if len(t.party) > 0:
#         fainted += t.party[0].hp == 0
#     if len(t.party) > 1:
#         fainted += t.party[1].hp == 0
#     return fainted


# def game_state_eval(s: GameState, depth):
#     mine = s.teams[0].active
#     opp = s.teams[1].active
#     return mine.hp / mine.max_hp - 3 * opp.hp / opp.max_hp - 0.3 * depth

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

    def select(self, node: "MCTSNode") -> "MCTSNode":
        """
        Selection phase: Navigate the tree to find the best node to expand.
        :param node: Current node to start selection from.
        :return: Selected node for expansion.
        """
        while node.children:  # Traverse the tree while child nodes exist
            node = max(
                node.children,
                key=lambda child: (
                    child.value / child.visits if child.visits > 0 else float('inf') +
                    self.exploration_constant * np.sqrt(np.log(node.visits + 1) / (child.visits + 1))
                )
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


class MCTSNode:
    """
    Represents a node in the MCTS search tree.
    """

    def __init__(self, g=None, parent=None, action=None):
        """
        Initialize the MCTS node.

        :param g: Game state at this node.
        :param parent: Parent node in the tree.
        :param action: Action taken to reach this node.
        """
        self.g = g  # GameState
        self.parent = parent  # Parent node
        self.children = []  # Child nodes
        self.action = action  # Action that led to this node
        self.visits = 0  # Number of visits
        self.value = 0.0  # Cumulative reward


def n_fainted(team: PkmTeam) -> int:
    """
    Count the number of fainted Pokémon in a team.
    :param team: Pokémon team.
    :return: Number of fainted Pokémon.
    """
    return sum(pkm.hp == 0 for pkm in [team.active] + team.party[:2])


def game_state_eval(g: GameState, depth: int) -> float:
    """
    Evaluate a game state to assign a reward.
    :param g: Game state to evaluate.
    :param depth: Current depth of the tree.
    :return: Calculated reward value.
    """
    my_active = g.teams[0].active
    opp_active = g.teams[1].active

    # Components of evaluation:
    hp_difference = my_active.hp / my_active.max_hp - opp_active.hp / opp_active.max_hp
    fainted_difference = n_fainted(g.teams[0]) - n_fainted(g.teams[1])

    # Weighted reward combining components
    return (
        10 * fainted_difference + 5 * hp_difference - 0.3 * depth
    )
