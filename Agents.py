from copy import deepcopy
from typing import List
import numpy as np
from vgc.behaviour.BattlePolicies import BattlePolicy
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition
from vgc.engine.PkmBattleEnv import PkmTeam, PkmBattleEnv
from utils import N_ACTIONS

# class RandomAgent(BattlePolicy):
#     """
#     Agent that selects actions randomly.
#     """

#     def __init__(self, switch_probability: float = .15, n_moves: int = 4,
#                  n_switches: int = 2):
#         super().__init__()
#         self.n_actions: int = n_moves + n_switches
#         self.n_switches = n_switches
#         self.pi: List[float] = ([(1. - switch_probability) / n_moves] * n_moves) + (
#                 [switch_probability / n_switches] * n_switches)

#     def get_action(self, gamestate: GameState) -> int:
#         my_type = gamestate.teams[0].active.type
#         opponent_type = gamestate.teams[1].active.type
#         print(PkmType(my_type).name, PkmType(opponent_type).name)
#         if(TYPE_CHART_MULTIPLIER[opponent_type][my_type] == 2):
#             move = self.n_actions - self.n_switches + np.random.choice(self.n_switches)
#             print("superefficacia: ", TYPE_CHART_MULTIPLIER[opponent_type][my_type], " mossa: ", move)
#             return move 
#         return np.random.choice(self.n_actions, p=self.pi)
    
class Node:

    def __init__(self):
        self.action = None
        self.gamestate = None
        self.parent = None
        self.depth = 0
        self.eval = 0.0
    
class MinimaxAgent(BattlePolicy):

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def get_action(self, gamestate: PkmBattleEnv) -> int:  # gamestate: PkmBattleEnv
        root: Node = Node()
        root.gamestate = gamestate
        queue: List[Node] = [root]
        while len(queue) > 0 and queue[0].depth < self.max_depth:
            parent = queue.pop(0)
            # expand nodes of current parent
            for i in range(N_ACTIONS):
                for j in range(N_ACTIONS):
                    gamestate = deepcopy(parent.gamestate)
                    state, _, _, _, _ = gamestate.step([i, j])  # opponent select an invalid switch action
                    # our fainted increased, skip
                    if n_fainted(state[0].teams[0]) > n_fainted(parent.gamestate.teams[0]):
                        continue
                    # our opponent fainted increased, follow this decision
                    if n_fainted(state[0].teams[1]) > n_fainted(parent.gamestate.teams[1]):
                        action = i
                        while parent != root:
                            action = parent.action
                            parent = parent.parent
                        print(action)

                        return action
                    # continue tree traversal
                    node = Node()
                    node.parent = parent
                    node.depth = node.parent.depth + 1
                    node.action = i
                    node.gamestate = state[0]
                    queue.append(node)
        # no possible win outcomes, return arbitrary action
        if len(queue) == 0:
            return 0
        # return action with most potential
        best_node = max(queue, key=lambda n: game_state_eval(n.gamestate, n.depth))
        while best_node.parent != root:
            best_node = best_node.parent
            print(best_node.action)
        return best_node.action

def n_fainted(team: PkmTeam):
    fainted = 0
    fainted += team.active.hp == 0
    if len(team.party) > 0:
        fainted += team.party[0].hp == 0
    if len(team.party) > 1:
        fainted += team.party[1].hp == 0
    return fainted


def game_state_eval(state: GameState, depth):
    mine = state.teams[0].active
    opp = state.teams[1].active
    return mine.hp / mine.max_hp - 3 * opp.hp / opp.max_hp - 0.3 * depth
