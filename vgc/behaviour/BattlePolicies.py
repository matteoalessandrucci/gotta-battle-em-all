import tkinter
from copy import deepcopy
from threading import Thread, Event
from tkinter import CENTER, DISABLED, NORMAL
from types import CellType
from typing import List

import numpy as np
from customtkinter import CTk, CTkButton, CTkRadioButton, CTkLabel

from vgc.behaviour import BattlePolicy
from vgc.datatypes.Constants import DEFAULT_PKM_N_MOVES, DEFAULT_PARTY_SIZE, TYPE_CHART_MULTIPLIER, DEFAULT_N_ACTIONS
from vgc.datatypes.Objects import GameState, PkmTeam
from vgc.datatypes.Types import PkmStat, PkmType, WeatherCondition


class RandomPlayer(BattlePolicy):
    """
    Agent that selects actions randomly.
    """

    def __init__(self, switch_probability: float = .15, n_moves: int = DEFAULT_PKM_N_MOVES,
                 n_switches: int = DEFAULT_PARTY_SIZE):
        super().__init__()
        self.n_actions: int = n_moves + n_switches
        self.pi: List[float] = ([(1. - switch_probability) / n_moves] * n_moves) + (
                [switch_probability / n_switches] * n_switches)

    def get_action(self, g: GameState) -> int:
        return np.random.choice(self.n_actions, p=self.pi)


def estimate_damage(move_type: PkmType, pkm_type: PkmType, move_power: float, opp_pkm_type: PkmType,
                    attack_stage: int, defense_stage: int, weather: WeatherCondition) -> float:
    stab = 1.5 if move_type == pkm_type else 1.
    if (move_type == PkmType.WATER and weather == WeatherCondition.RAIN) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.SUNNY):
        weather = 1.5
    elif (move_type == PkmType.WATER and weather == WeatherCondition.SUNNY) or (
            move_type == PkmType.FIRE and weather == WeatherCondition.RAIN):
        weather = .5
    else:
        weather = 1.
    stage_level = attack_stage - defense_stage
    stage = (stage_level + 2.) / 2 if stage_level >= 0. else 2. / (np.abs(stage_level) + 2.)
    damage = TYPE_CHART_MULTIPLIER[move_type][opp_pkm_type] * stab * weather * stage * move_power
    return damage


class OneTurnLookahead(BattlePolicy):
    """
    Greedy heuristic based competition designed to encapsulate a greedy strategy that prioritizes damage output.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def get_action(self, g: GameState):
        # get weather condition
        weather = g.weather.condition

        # get my pkms
        my_team = g.teams[0]
        my_active = my_team.active
        my_attack_stage = my_team.stage[PkmStat.ATTACK]

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # get most damaging move from my active pkm
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active_type,
                                          my_attack_stage, opp_defense_stage, weather))

        return int(np.argmax(damage))  # use active pkm best damaging move


def match_up_eval(my_pkm_type: PkmType, opp_pkm_type: PkmType, opp_moves_type: List[PkmType]) -> float:
    # determine defensive match up
    defensive_match_up = 0.0
    for mtype in opp_moves_type + [opp_pkm_type]:
        defensive_match_up = max(TYPE_CHART_MULTIPLIER[mtype][my_pkm_type], defensive_match_up)
    return defensive_match_up


class TypeSelector(BattlePolicy):
    """
    Type Selector is a variation upon the One Turn Lookahead competition that utilizes a short series of if-else
    statements in its decision-making.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def get_action(self, g: GameState):
        # get weather condition
        weather = g.weather.condition

        # get my pokémon
        my_team = g.teams[0]
        my_active = my_team.active
        my_party = my_team.party
        my_attack_stage = my_team.stage[PkmStat.ATTACK]

        # get opp team
        opp_team = g.teams[1]
        opp_active = opp_team.active
        opp_active_type = opp_active.type
        opp_defense_stage = opp_team.stage[PkmStat.DEFENSE]

        # get most damaging move from my active pokémon
        damage: List[float] = []
        for move in my_active.moves:
            damage.append(estimate_damage(move.type, my_active.type, move.power, opp_active_type,
                                          my_attack_stage, opp_defense_stage, weather))
        move_id = int(np.argmax(damage))

        #  If this damage is greater than the opponents current health we knock it out
        if damage[move_id] >= opp_active.hp:
            return move_id

        # If not, check if are a favorable match. If we are lets give maximum possible damage.
        if match_up_eval(my_active.type, opp_active.type, list(map(lambda m: m.type, opp_active.moves))) <= 1.0:
            return move_id

        # If we are not switch to the most favorable match up
        match_up: List[float] = []
        not_fainted = False
        for pkm in my_party:
            if pkm.hp == 0.0:
                match_up.append(2.0)
            else:
                not_fainted = True
                match_up.append(
                    match_up_eval(pkm.type, opp_active.type, list(map(lambda m: m.type, opp_active.moves))))

        if not_fainted:
            return int(np.argmin(match_up)) + 4

        # If our party has no non fainted pkm, lets give maximum possible damage with current active
        return move_id


class BFSNode:

    def __init__(self):
        self.a = None
        self.g = None
        self.parent = None
        self.depth = 0
        self.eval = 0.0


def n_fainted(t: PkmTeam):
    fainted = 0
    fainted += t.active.hp == 0
    if len(t.party) > 0:
        fainted += t.party[0].hp == 0
    if len(t.party) > 1:
        fainted += t.party[1].hp == 0
    return fainted


def game_state_eval(s: GameState, depth):
    mine = s.teams[0].active
    opp = s.teams[1].active
    return mine.hp / mine.max_hp - 3 * opp.hp / opp.max_hp - 0.3 * depth


class BreadthFirstSearch(BattlePolicy):
    """
    Basic tree search algorithm that traverses nodes in level order until it finds a state in which the current opponent
    Pokémon is fainted. As a non-adversarial algorithm, the competition selfishly assumes that the opponent uses
    "force skip" (by selecting an invalid switch action).
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def get_action(self, g) -> int:  # g: PkmBattleEnv
        root: BFSNode = BFSNode()
        root.g = g
        node_queue: List[BFSNode] = [root]
        while len(node_queue) > 0 and node_queue[0].depth < self.max_depth:
            current_parent = node_queue.pop(0)
            # expand nodes of current parent
            for i in range(DEFAULT_N_ACTIONS):
                g = deepcopy(current_parent.g)
                s, _, _, _, _ = g.step([i, 99])  # opponent select an invalid switch action
                # our fainted increased, skip
                if n_fainted(s[0].teams[0]) > n_fainted(current_parent.g.teams[0]):
                    continue
                # our opponent fainted increased, follow this decision
                if n_fainted(s[0].teams[1]) > n_fainted(current_parent.g.teams[1]):
                    a = i
                    while current_parent != root:
                        a = current_parent.a
                        current_parent = current_parent.parent
                    return a
                # continue tree traversal
                node = BFSNode()
                node.parent = current_parent
                node.depth = node.parent.depth + 1
                node.a = i
                node.g = s[0]
                node_queue.append(node)
        # no possible win outcomes, return arbitrary action
        if len(node_queue) == 0:
            return 0
        # return action with most potential
        best_node = max(node_queue, key=lambda n: game_state_eval(n.g, n.depth))
        while best_node.parent != root:
            best_node = best_node.parent
        return best_node.a


class Minimax(BattlePolicy):
    """
    Tree search algorithm that deals with adversarial paradigms by assuming the opponent acts in their best interest.
    Each node in this tree represents the worst case scenario that would occur if the player had chosen a specific
    choice.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def get_action(self, g) -> int:  # g: PkmBattleEnv
        root: BFSNode = BFSNode()
        root.g = g
        node_queue: List[BFSNode] = [root]
        while len(node_queue) > 0 and node_queue[0].depth < self.max_depth:
            current_parent = node_queue.pop(0)
            # expand nodes of current parent
            for i in range(DEFAULT_N_ACTIONS):
                for j in range(DEFAULT_N_ACTIONS):
                    g = deepcopy(current_parent.g)
                    s, _, _, _, _ = g.step([i, j])  # opponent select an invalid switch action
                    # our fainted increased, skip
                    if n_fainted(s[0].teams[0]) > n_fainted(current_parent.g.teams[0]):
                        continue
                    # our opponent fainted increased, follow this decision
                    if n_fainted(s[0].teams[1]) > n_fainted(current_parent.g.teams[1]):
                        a = i
                        while current_parent != root:
                            a = current_parent.a
                            current_parent = current_parent.parent
                        return a
                    # continue tree traversal
                    node = BFSNode()
                    node.parent = current_parent
                    node.depth = node.parent.depth + 1
                    node.a = i
                    node.g = s[0]
                    node_queue.append(node)
        # no possible win outcomes, return arbitrary action
        if len(node_queue) == 0:
            return 0
        # return action with most potential
        best_node = max(node_queue, key=lambda n: game_state_eval(n.g, n.depth))
        while best_node.parent != root:
            best_node = best_node.parent
        return best_node.a


class PrunedBFS(BattlePolicy):
    """
    Utilize domain knowledge as a cost-cutting measure by making modifications to the Breadth First Search competition.
    We do not simulate any actions that involve using a damaging move with a resisted type, nor does it simulate any
    actions that involve switching to a Pokémon with a subpar type match up. Additionally, rather than selfishly
    assuming the opponent skips their turn in each simulation, the competition assumes its opponent is a
    One Turn Lookahead competition.
    Source: http://www.cig2017.com/wp-content/uploads/2017/08/paper_87.pdf
    """

    def __init__(self, max_depth: int = 4):
        self.core_agent: BattlePolicy = OneTurnLookahead()
        self.max_depth = max_depth

    def get_action(self, g) -> int:  # g: PkmBattleEnv
        root: BFSNode = BFSNode()
        root.g = g
        node_queue: List[BFSNode] = [root]
        while len(node_queue) > 0 and node_queue[0].depth < self.max_depth:
            current_parent = node_queue.pop(0)
            # assume opponent follows just the OneTurnLookahead strategy, which is more greedy in damage
            o: GameState = deepcopy(current_parent.g)
            # opponent must see the teams swapped
            o.teams = (o.teams[1], o.teams[0])
            j = self.core_agent.get_action(o)
            # expand nodes
            for i in range(DEFAULT_N_ACTIONS):
                g = deepcopy(current_parent.g)
                my_team = g.teams[0]
                my_active = my_team.active
                opp_team = g.teams[1]
                opp_active = opp_team.active
                # skip traversing tree with non very effective moves
                if i < 4 and TYPE_CHART_MULTIPLIER[my_active.moves[i].type][opp_active.type] < 1.0:
                    continue
                # skip traversing tree with switches to Pokémon that are a bad type match against opponent active
                elif i >= 4:
                    p = i - DEFAULT_N_ACTIONS
                    for move in opp_active.moves:
                        if move.power > 0.0 and TYPE_CHART_MULTIPLIER[move.type][my_team.party[p].type] > 1.0:
                            continue
                s, _, _, _, _ = g.step([i, j])
                # our fainted increased, skip
                if n_fainted(s[0].teams[0]) > n_fainted(current_parent.g.teams[0]):
                    continue
                # our opponent fainted increased, follow this decision
                if n_fainted(s[0].teams[1]) > n_fainted(current_parent.g.teams[1]):
                    a = i
                    while current_parent != root:
                        a = current_parent.a
                        current_parent = current_parent.parent
                    return a
                # continue tree traversal
                node = BFSNode()
                node.parent = current_parent
                node.depth = node.parent.depth + 1
                node.a = i
                node.g = s[0]
                node_queue.append(node)
        # no possible win outcomes, return arbitrary action
        if len(node_queue) == 0:
            a = self.core_agent.get_action(g)
            return a
        # return action with most potential
        best_node = max(node_queue, key=lambda n: game_state_eval(n.g, n.depth))
        while best_node.parent != root:
            best_node = best_node.parent
        return best_node.a


class TunedTreeTraversal(BattlePolicy):
    """
    Agent inspired on PrunedBFS. Assumes opponent is a TypeSelector. It traverses only to states after using moves
    recommended by a TypeSelector agent and non-damaging moves.
    """

    def __init__(self, max_depth: int = 4):
        self.core_agent: BattlePolicy = TypeSelector()
        self.max_depth = max_depth

    def get_action(self, g: GameState) -> int:  # g: PkmBattleEnv
        root: BFSNode = BFSNode()
        root.g = g
        node_queue: List[BFSNode] = [root]
        while len(node_queue) > 0 and node_queue[0].depth < self.max_depth:
            current_parent = node_queue.pop(0)
            # assume opponent follows just the TypeSelector strategy, which is more greedy in damage
            o: GameState = deepcopy(current_parent.g)
            # opponent must see the teams swapped
            o.teams = (o.teams[1], o.teams[0])
            j = self.core_agent.get_action(o)
            # expand nodes with TypeSelector strategy plus non-damaging moves
            for i in [self.core_agent.get_action(current_parent.g)] + [i for i, m in enumerate(
                    current_parent.g.teams[0].active.moves) if m.power == 0.]:
                g = deepcopy(current_parent.g)
                s, _, _, _, _ = g.step([i, j])
                # our fainted increased, skip
                if n_fainted(s[0].teams[0]) > n_fainted(current_parent.g.teams[0]):
                    continue
                # our opponent fainted increased, follow this decision
                if n_fainted(s[0].teams[1]) > n_fainted(current_parent.g.teams[1]):
                    a = i
                    while current_parent != root:
                        a = current_parent.a
                        current_parent = current_parent.parent
                    return a
                # continue tree traversal
                node = BFSNode()
                node.parent = current_parent
                node.depth = node.parent.depth + 1
                node.a = i
                node.g = s[0]
                node_queue.append(node)
        # no possible win outcomes, return arbitrary action
        if len(node_queue) == 0:
            a = self.core_agent.get_action(g)
            return a
        # return action with most potential
        best_node = max(node_queue, key=lambda n: game_state_eval(n.g, n.depth))
        while best_node.parent != root:
            best_node = best_node.parent
        return best_node.a


class TerminalPlayer(BattlePolicy):
    """
    Terminal interface.
    """

    def get_action(self, g: GameState) -> int:
        print('~ Actions ~')
        for i, a in enumerate(g.teams[0].active.moves):
            print(i, '->', a)
        for i, a in enumerate(g.teams[0].party):
            print(i + DEFAULT_PKM_N_MOVES, '-> Switch to ', a)
        while True:
            try:
                m = int(input('Select Action: '))
                if 0 < m < DEFAULT_N_ACTIONS:
                    break
                else:
                    print('Invalid action. Select again.')
            except:
                print('Invalid action. Select again.')
        print()
        return m


def run_tk(event, value_wrapper, game_state_wrapper):
    app = GUIPlayer.App()
    app.event = event
    app.value_wrapper = value_wrapper
    app.game_state_wrapper = game_state_wrapper
    app.update_game_state()
    app.mainloop()


def disable_event():
    pass


class GUIPlayer(BattlePolicy):
    """
    Graphical interface.
    """

    class App(CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.title("VGC Battle GUI")
            #self.iconbitmap(r"vgc/ux/vgc_v2_01_Uw5_icon.ico")
            self.geometry("650x200")
            self.protocol("WM_DELETE_WINDOW", disable_event)
            CTkLabel(self, text="Actions").pack(anchor='w')
            self.button = CTkButton(self, text="Select Action", state=DISABLED, command=self.button_callback)
            self.button.place(relx=0.5, rely=0.9, anchor=CENTER)
            self.radio_var = tkinter.IntVar(value=0)  # Create a variable for strings, and initialize the variable
            self.buttons = []
            for i in range(DEFAULT_N_ACTIONS):
                button = CTkRadioButton(self, text=f"{i}", state=DISABLED, variable=self.radio_var, value=i)
                button.pack(anchor='w')
                self.buttons += [button]
            self.event = None
            self.value_wrapper = None
            self.game_state_wrapper = None

        def button_callback(self):
            self.button.configure(state=DISABLED)
            for button in self.buttons:
                button.configure(state=DISABLED)
            self.value_wrapper.cell_contents = self.radio_var.get()
            self.event.set()

        def update_game_state(self):
            g = self.game_state_wrapper.cell_contents
            if g is not None and not isinstance(g, int):
                self.button.configure(state=NORMAL)
                for i, button in enumerate(self.buttons):
                    if i < DEFAULT_PKM_N_MOVES:
                        button.configure(state=NORMAL, text=f"{g.teams[0].active.moves[i]}")
                    else:
                        button.configure(state=NORMAL, text=f"Switch to {g.teams[0].party[i - DEFAULT_PKM_N_MOVES]}")
                self.game_state_wrapper.cell_contents = None
            self.after(1000, self.update_game_state)

    def __init__(self):
        self.event = Event()
        self.value_wrapper = CellType(0)
        self.game_state_wrapper = CellType(0)
        self.thd = Thread(target=run_tk, args=(self.event, self.value_wrapper, self.game_state_wrapper))
        self.thd.daemon = True
        self.thd.start()

    def get_action(self, g: GameState) -> int:
        self.game_state_wrapper.cell_contents = g
        self.event.wait()
        self.event.clear()
        return self.value_wrapper.cell_contents
