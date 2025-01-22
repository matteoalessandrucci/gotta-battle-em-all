import time
from copy import deepcopy
import numpy as np
from utils import MCTSNode, evaluate_game_state
from vgc.behaviour.BattlePolicies import BattlePolicy
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Types import PkmType
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER


class MonteCarloAgent(BattlePolicy):
    """
    Monte Carlo Tree Search (MCTS) agent for decision-making in Pokémon battles.

    Attributes:
        max_simulations: Maximum number of simulations allowed at the start of the game.
        min_simulations: Minimum number of simulations allowed toward the end of the game.
        min_exploration_constant: Minimum exploration constant (used in UCB formula) at the start of the game.
        max_exploration_constant: Maximum exploration constant toward the end of the game.
        time_limit: Optional time limit in seconds for each decision.
        simulations: Current number of simulations, dynamically adapted during the game.
        exploration_constant: Current exploration constant, dynamically adapted during the game.
    """

    def __init__(self, max_simulations: int = 50, min_simulations: int = 30, 
                 min_exploration_constant: float = 1.2, max_exploration_constant: float = 1.4,
                 time_limit: float = None):
        """
        Initialize the MCTS agent with dynamic parameter adaptation.
        """
        self.max_simulations = max_simulations
        self.min_simulations = min_simulations
        self.min_exploration_constant = min_exploration_constant
        self.max_exploration_constant = max_exploration_constant
        self.time_limit = time_limit
        self.simulations = max_simulations
        self.exploration_constant = min_exploration_constant

    def get_action(self, g) -> int:
        """
        Determine the best action to take from the current game state using MCTS.
        Dynamically adjusts the number of simulations and exploration constant.

        :param g: The current game state.
        :return: The best action to take, based on MCTS.
        """
        self.adapt_parameters(g)  # Adapt parameters based on game progress
        root = MCTSNode(g=g)
        start_time = time.time()

        # Perform simulations within the defined constraints
        for _ in range(self.simulations):
            current_time = time.time()
            if self.time_limit and (current_time - start_time) >= self.time_limit:
                print(f"Time limit reached: {current_time - start_time:.2f} seconds")
                break
            
            selected_node = self.select(root)  # Selection phase
            
            if self.is_terminal(selected_node):  # Skip terminal states
                continue
            
            expanded_node = self.expand(selected_node)  # Expansion phase
            
            if not expanded_node.children:  # If expansion fails, skip this simulation
                continue
            
            reward = self.simulate(expanded_node)  # Simulation phase
            self.backpropagate(expanded_node, reward)  # Backpropagation phase

        # Select the best action based on the highest visit count
        if not root.children:  # Fallback to random action if no children exist
            return np.random.choice(range(6))
        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.action


    def adapt_parameters(self, g: GameState):
        """
        Dynamically adjust the number of simulations and exploration constant based on game progress.
        Progress is calculated as the fraction of total Pokémon fainted in both teams.
        """
        my_remaining = sum(pkm.hp > 0 for pkm in g.teams[0].party + [g.teams[0].active])
        opp_remaining = sum(pkm.hp > 0 for pkm in g.teams[1].party + [g.teams[1].active])
        total_pokemon = len(g.teams[0].party) + len(g.teams[1].party) + 2
        progress = (total_pokemon - (my_remaining + opp_remaining)) / total_pokemon

        self.simulations = int(self.min_simulations + (self.max_simulations - self.min_simulations) * progress)
        self.exploration_constant = (
            self.max_exploration_constant
            - (self.max_exploration_constant - self.min_exploration_constant) * progress
        )

    def select(self, node: "MCTSNode") -> "MCTSNode":
        """
        Selection phase: Traverse the tree using the UCB formula to find the best node for expansion.
        """
        while node.children and not self.is_terminal(node):
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
        if not node.children and not self.is_terminal(node):
            for action in range(6):
                child_g = deepcopy(node.g)
                opponent_action = np.random.choice(6) #Random opponent action use (self.heuristic_action(child_g.teams[1])) to test against opponents that use heuristic
                child_g.step([action, opponent_action])
                child_node = MCTSNode(g=child_g, parent=node, action=action)
                node.children.append(child_node)
        
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
                self.heuristic_action(sim_g.teams[0]),  # Action based on heuristic for agent use (np.random.choice(6)) for random action for agent
                np.random.choice(6)  # Random action for opponent use (self.heuristic_action(sim_g.teams[1]) ) to simulate an action based on heuristic for opponent
            ]
            _, _, t, _, _ = sim_g.step(actions)
            turns += 1

        return evaluate_game_state(sim_g, depth=turns)
    
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
        Evaluate the desirability of an action based on type advantage and damage.
        """
        simulated_team = deepcopy(team)
        opponent_type = team.active.type.value
        move_type = self.get_move_type(action).value

        advantage = TYPE_CHART_MULTIPLIER[move_type][opponent_type]
        damage = action * 5 * advantage
        simulated_team.active.hp -= damage

        return simulated_team.active.hp / simulated_team.active.max_hp

    def get_move_type(self, action: int):
        """
        Map action indices to move types.
        """
        move_types = [
            PkmType.NORMAL,
            PkmType.FIRE,
            PkmType.WATER,
            PkmType.GRASS,
            PkmType.ELECTRIC,
            PkmType.ICE,
        ]
        return move_types[action % len(move_types)]


    def is_terminal(self, node: "MCTSNode") -> bool:
        """
        Check if the node represents a terminal state (game over).
        """
        return all(pkm.hp == 0 for pkm in node.g.teams[0].party + [node.g.teams[0].active]) or \
               all(pkm.hp == 0 for pkm in node.g.teams[1].party + [node.g.teams[1].active])