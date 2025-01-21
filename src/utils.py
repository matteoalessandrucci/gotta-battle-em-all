import sys
import os
sys.path.append(os.path.join(sys.path[0], ".."))

from vgc.datatypes.Objects import GameState
from vgc.engine.PkmBattleEnv import PkmTeam


from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER

class MinimaxNode:
    """
    Represents a node in the Minimax game tree.
    """
    def __init__(self, state, action=None, parent=None, depth=0):
        self.state = state         # Game state at this node
        self.action = action       # Action taken to reach this state
        self.parent = parent       # Parent node
        self.depth = depth         # Depth of the node in the tree
        self.children = []         # Child nodes
        self.eval_value = None     # Evaluation score of this node

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
    agent = game_state.teams[0]
    opp = game_state.teams[1]

    # Difference between HP (weighted)
    score = agent.active.hp / agent.active.max_hp - 3 * opp.active.hp / opp.active.max_hp

    # Difference between fainted Pokèmon (weighted)
    score += 15 * (3 - n_fainted(opp) - (3 - n_fainted(agent)))

    # type effictiveness bonus 
    score += TYPE_CHART_MULTIPLIER[agent.active.type][opp.active.type] * 10

    # Almost KO pokemon penalty
    if agent.active.hp / agent.active.max_hp < 0.2:
        score -= 10

    # Depth penalty for the node
    return score - 0.1 * depth

def evaluate_game_state(game_state: GameState) -> float:
        agent = game_state.teams[0]
        opp = game_state.teams[1]
        score = sum(pkm.hp for pkm in agent.party + [agent.active]) - sum(
            pkm.hp for pkm in opp.party + [opp.active]
        )
        score += 100 * (len(agent.party) - len(opp.party))
        score += 50 * (
            len([pkm for pkm in opp.party if pkm.hp <= 0])
            - len([pkm for pkm in agent.party if pkm.hp <= 0])
        )
        score += TYPE_CHART_MULTIPLIER[agent.active.type][opp.active.type] * 10
        return score