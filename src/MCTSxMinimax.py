from typing import List
import copy
from copy import deepcopy
import numpy as np
import os
import sys
import math
from utils import MinimaxNode, MCTSNode, minimax, evaluate_game_state_MCTS

sys.path.append(os.path.join(sys.path[0], ".."))

from vgc.behaviour.BattlePolicies import BattlePolicy
from vgc.datatypes.Objects import GameState

class MCTSxMinimax(BattlePolicy):
    def __init__(self, n_moves: int = 4, n_switches: int = 2, max_depth: int = 2, simulations: int = 50, exploration_constant: float = 1.4, mcts_depth: int = 2):
        super().__init__()
        self.n_moves = n_moves
        self.n_switches = n_switches
        self.n_actions = n_moves + n_switches
        self.max_depth = max_depth
        self.simulations = simulations
        self.exploration_constant = exploration_constant
        self.mcts_depth = mcts_depth  # Depth for MCTS simulation

    def get_action(self, game_state: GameState) -> int:
        # Execute MCTS
        mcts_root = self.run_mcts(game_state, self.mcts_depth)

        # Order the children nodes of root based on MCTS score
        best_mcts_actions = sorted(
            mcts_root.children,
            key=lambda n: n.value / n.visits if n.visits > 0 else float("-inf"),
            reverse=True,
        )

        best_action = None
        best_score = -math.inf

        for node in best_mcts_actions:
            # Minimax evaluation
            root_node = MinimaxNode(node.g)  # minimax start node
            eval_value, _ = minimax(
                root_node, 0, self.max_depth, -math.inf, math.inf, True
            )

            # choose the best scoring action according to Minimax
            if eval_value > best_score:
                best_score = eval_value
                best_action = node.action

        return best_action
    
    def _is_terminal(self, game_state: GameState) -> bool:
        return all(
            pkm.hp <= 0
            for pkm in game_state.teams[0].party + [game_state.teams[0].active]
        ) or all(
            pkm.hp <= 0
            for pkm in game_state.teams[1].party + [game_state.teams[1].active]
        )

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
                    (child.value / child.visits if child.visits > 0 else float("inf"))
                    + self.exploration_constant
                    * np.sqrt(np.log(node.visits + 1) / (child.visits + 1))
                ),
            )
        return node

    def expand(self, node: MCTSNode, depth: int) -> MCTSNode:
        if not node.children or node.g.turn < depth:
            for action in range(self.n_actions):
                child_g = deepcopy(node.g)
                child_g.step(
                    [action, np.random.choice(self.n_actions)]
                )  # simulates a random action for the opponent
                child_node = MCTSNode(g=child_g, parent=node, action=action)
                node.children.append(child_node)

        if node.children:
            return np.random.choice(node.children)
        else:
            return node  # Returns current node if there are no children

    def simulate(self, node: MCTSNode) -> float:
        sim_g = deepcopy(node.g)
        t = False
        max_turns = 10
        turns = 0
        while not t and turns < max_turns:
            actions = [np.random.choice(self.n_actions) for _ in range(2)]
            _, _, t, _, _ = sim_g.step(actions)
            turns += 1
        return evaluate_game_state_MCTS(sim_g, turns)

    def backpropagate(self, node: MCTSNode, reward: float):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
