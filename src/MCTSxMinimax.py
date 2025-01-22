from typing import List
import copy
from copy import deepcopy
import numpy as np
import os
import sys
import math
import time
from utils import *
from vgc.datatypes.Types import PkmType

sys.path.append(os.path.join(sys.path[0], ".."))

from vgc.behaviour.BattlePolicies import BattlePolicy
from vgc.datatypes.Objects import GameState

class MCTSxMinimax(BattlePolicy):
    def __init__(
        self,
        n_moves: int = 4,
        n_switches: int = 2,
        max_depth: int = 2,

        mcts_depth: int = 2,
        max_simulations: int = 50, min_simulations: int = 30, 
                 min_exploration_constant: float = 1.2, max_exploration_constant: float = 1.5,
                 time_limit: float = None
    ):
        super().__init__()
        self.n_moves = n_moves
        self.n_switches = n_switches
        self.n_actions = n_moves + n_switches
        self.max_depth = max_depth
        self.mcts_depth = mcts_depth  # Depth for MCTS simulation
        self.max_simulations = max_simulations
        self.min_simulations = min_simulations
        self.min_exploration_constant = min_exploration_constant
        self.max_exploration_constant = max_exploration_constant
        self.time_limit = time_limit
        self.simulations = max_simulations
        self.exploration_constant = min_exploration_constant

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

        # Ensure best_action is not None
        if best_action is None:
            best_action = np.random.choice(range(6))

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
        self.adapt_parameters_inverse(game_state)  # Adapt parameters dynamically

        root = MCTSNode(g=game_state)
        start_time = time.time()

        # Perform simulations
        for _ in range(self.simulations):
            current_time = time.time()  # Controllo più frequente
            if self.time_limit and (current_time - start_time) >= self.time_limit:
                print(f"Time limit reached: {current_time - start_time:.2f} seconds")
                break
            
            selected_node = self.select(root)
            
            if self.time_limit and (time.time() - start_time) >= self.time_limit:
                print(f"Time limit reached after select: {time.time() - start_time:.2f} seconds")
                break
            
            if self.is_terminal(selected_node):  # Skip simulation for terminal nodes
                continue
            
            expanded_node = self.expand(selected_node)
            
            if self.time_limit and (time.time() - start_time) >= self.time_limit:
                print(f"Time limit reached after expand: {time.time() - start_time:.2f} seconds")
                break
            
            if not expanded_node.children:  # If expansion fails, skip this simulation
                continue
            
            reward = self.simulate(expanded_node)
            
            if self.time_limit and (time.time() - start_time) >= self.time_limit:
                print(f"Time limit reached after simulate: {time.time() - start_time:.2f} seconds")
                break
            
            self.backpropagate(expanded_node, reward)

        # Select the best action based on the highest visit count
        if not root.children:  # If no children were expanded, fallback to random action
            return np.random.choice(range(6))
        
        best_child = max(root.children, key=lambda n: n.visits)
        return root

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
        self.simulations = int(self.min_simulations + (self.max_simulations - self.min_simulations) * progress)
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

        #return game_state_eval(sim_g, depth=turns)
        return self._evaluate_gamestate(sim_g)

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




    def is_terminal(self, node: "MCTSNode") -> bool:
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



# from typing import List
# import copy
# from copy import deepcopy
# import numpy as np
# import os
# import sys
# import math
# from utils import MinimaxNode, MCTSNode, minimax, evaluate_game_state_MCTS

# sys.path.append(os.path.join(sys.path[0], ".."))

# from vgc.behaviour.BattlePolicies import BattlePolicy
# from vgc.datatypes.Objects import GameState

# class MCTSxMinimax(BattlePolicy):
#     def __init__(
#         self,
#         n_moves: int = 4,
#         n_switches: int = 2,
#         max_depth: int = 2,
#         simulations: int = 50,
#         exploration_constant: float = 1.4,
#         mcts_depth: int = 2,
#     ):
#         super().__init__()
#         self.n_moves = n_moves
#         self.n_switches = n_switches
#         self.n_actions = n_moves + n_switches
#         self.max_depth = max_depth
#         self.simulations = simulations
#         self.exploration_constant = exploration_constant
#         self.mcts_depth = mcts_depth  # Depth for MCTS simulation

#     def get_action(self, game_state: GameState) -> int:
#         # Execute MCTS
#         mcts_root = self.run_mcts(game_state, self.mcts_depth)

#         # Order the children nodes of root based on MCTS score
#         best_mcts_actions = sorted(
#             mcts_root.children,
#             key=lambda n: n.value / n.visits if n.visits > 0 else float("-inf"),
#             reverse=True,
#         )

#         best_action = None
#         best_score = -math.inf

#         for node in best_mcts_actions:
#             # Minimax evaluation
#             root_node = MinimaxNode(node.g)  # minimax start node
#             eval_value, _ = minimax(
#                 root_node, 0, self.max_depth, -math.inf, math.inf, True
#             )

#             # choose the best scoring action according to Minimax
#             if eval_value > best_score:
#                 best_score = eval_value
#                 best_action = node.action

#         return best_action
    
#     def _is_terminal(self, game_state: GameState) -> bool:
#         return all(
#             pkm.hp <= 0
#             for pkm in game_state.teams[0].party + [game_state.teams[0].active]
#         ) or all(
#             pkm.hp <= 0
#             for pkm in game_state.teams[1].party + [game_state.teams[1].active]
#         )

#     def run_mcts(self, game_state: GameState, depth: int):
#         root = MCTSNode(g=game_state)
#         for i in range(self.simulations):
#             selected_node = self.select(root)
#             expanded_node = self.expand(selected_node, depth)
#             reward = self.simulate(expanded_node)
#             self.backpropagate(expanded_node, reward)
#         return root

#     def select(self, node: MCTSNode) -> MCTSNode:
#         while node.children:
#             node = max(
#                 node.children,
#                 key=lambda child: (
#                     (child.value / child.visits if child.visits > 0 else float("inf"))
#                     + self.exploration_constant
#                     * np.sqrt(np.log(node.visits + 1) / (child.visits + 1))
#                 ),
#             )
#         return node

#     def expand(self, node: MCTSNode, depth: int) -> MCTSNode:
#         if not node.children or node.g.turn < depth:
#             for action in range(self.n_actions):
#                 child_g = deepcopy(node.g)
#                 child_g.step(
#                     [action, np.random.choice(self.n_actions)]
#                 )  # simulates a random action for the opponent
#                 child_node = MCTSNode(g=child_g, parent=node, action=action)
#                 node.children.append(child_node)

#         if node.children:
#             return np.random.choice(node.children)
#         else:
#             return node  # Returns current node if there are no children

#     def simulate(self, node: MCTSNode) -> float:
#         sim_g = deepcopy(node.g)
#         t = False
#         max_turns = 10
#         turns = 0
#         while not t and turns < max_turns:
#             actions = [np.random.choice(self.n_actions) for _ in range(2)]
#             _, _, t, _, _ = sim_g.step(actions)
#             turns += 1
#         return evaluate_game_state_MCTS(sim_g, turns)

#     def backpropagate(self, node: MCTSNode, reward: float):
#         while node:
#             node.visits += 1
#             node.value += reward
#             node = node.parent