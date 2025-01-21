from typing import List
import numpy as np
from vgc.behaviour.BattlePolicies import BattlePolicy
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition


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