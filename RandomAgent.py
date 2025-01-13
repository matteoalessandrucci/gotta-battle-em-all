from typing import List
import numpy as np
from vgc.behaviour.BattlePolicies import BattlePolicy
from vgc.datatypes.Objects import GameState
from vgc.datatypes.Constants import TYPE_CHART_MULTIPLIER
from vgc.datatypes.Types import PkmType
import copy

class Agent(BattlePolicy):
    def __init__(self, switch_probability: float = 0.15, n_moves: int = 4, n_switches: int = 2):
        super().__init__()
        self.n_moves = n_moves
        self.n_switches = n_switches
        self.n_actions = n_moves + n_switches
        self.pi = self._calculate_action_probabilities(switch_probability)

    def _calculate_action_probabilities(self, switch_probability: float) -> List[float]:
        """Calcola la distribuzione di probabilità tra attacchi e cambi."""
        attack_prob = (1. - switch_probability) / self.n_moves
        switch_prob = switch_probability / self.n_switches
        return [attack_prob] * self.n_moves + [switch_prob] * self.n_switches

    def get_action(self, game_state: GameState) -> int:
        """Determina la miglior azione da intraprendere usando Monte Carlo e Minimax."""
        n = 10
        possible_actions = list(range(self.n_actions))
        montecarlo_values = {action: self._montecarlo_rollout(game_state, action) for action in possible_actions}
        best_actions = sorted(possible_actions, key=lambda a: montecarlo_values[a], reverse=True)[:n]

        best_move, best_score = None, float('-inf')
        for action in best_actions:
            depth = self._calculate_dynamic_depth(game_state)
            score = self._minimax_evaluation(game_state, action, depth, float('-inf'), float('inf'))
            if score > best_score:
                best_move, best_score = action, score
        return best_move

    def _montecarlo_rollout(self, game_state: GameState, action: int, simulations: int = 10, max_depth: int = 6) -> float:
        """Simula una serie di partite a partire da un'azione iniziale."""
        total_score = 0.0
        for _ in range(simulations):
            sim_state = copy.deepcopy(game_state)
            self._apply_action(sim_state, action)
            current_depth = 1
            while current_depth < max_depth and not self._is_terminal(sim_state):
                next_action = self._select_prioritized_action(sim_state)
                self._apply_action(sim_state, next_action)
                current_depth += 1
            total_score += self._evaluate_gamestate(sim_state)
        return total_score / simulations

    def _apply_action(self, game_state: GameState, action: int):
        """Applica un'azione al game state (attacco o cambio)."""
        if action < self.n_moves:
            self._apply_move(game_state, action)
        else:
            self._apply_switch(game_state, action - self.n_moves)

    def _apply_move(self, game_state: GameState, action: int):
        """Esegue una mossa di attacco."""
        best_move = self._choose_best_move(game_state)
        game_state.teams[1].active.hp -= best_move.fixed_damage

    def _apply_switch(self, game_state: GameState, switch_index: int):
        """Esegue un cambio di Pokémon."""
        game_state.teams[0].active = game_state.teams[0].party[switch_index]

    def _choose_best_move(self, game_state: GameState):
        """Seleziona la mossa migliore basandosi sull'efficacia di tipo."""
        return max(game_state.teams[0].active.moves, key=lambda move: self._get_move_effectiveness(move, game_state.teams[1].active))

    def _get_move_effectiveness(self, move, opponent_pokemon):
        """Calcola il moltiplicatore di efficacia per una mossa."""
        return TYPE_CHART_MULTIPLIER[move.type][opponent_pokemon.type]

    def _select_prioritized_action(self, game_state: GameState) -> int:
        """Seleziona un'azione casuale tra quelle meno rischiose."""
        safe_actions = [
            action for action in range(self.n_actions)
            if not self._is_risky_action(game_state, action)
        ]
        return np.random.choice(safe_actions) if safe_actions else np.random.choice(range(self.n_actions))

    def _is_risky_action(self, game_state: GameState, action: int) -> bool:
        """Determina se un'azione è rischiosa."""
        if action >= self.n_moves:
            switch_pokemon = game_state.teams[0].party[action - self.n_moves]
            opponent_pokemon = game_state.teams[1].active
            if switch_pokemon.hp <= 30 or TYPE_CHART_MULTIPLIER[switch_pokemon.type][opponent_pokemon.type] < 1.0:
                return True
            if switch_pokemon.status != "PkmStatus.NONE":
                return True
        return False

    def _evaluate_gamestate(self, game_state: GameState) -> float:
        """Valuta lo stato attuale del gioco."""
        ally = game_state.teams[0]
        opp = game_state.teams[1]
        score = sum(pkm.hp for pkm in ally.party + [ally.active]) - sum(pkm.hp for pkm in opp.party + [opp.active])
        score += 100 * (len(ally.party) - len(opp.party))
        score += 50 * (len([pkm for pkm in opp.party if pkm.hp <= 0]) - len([pkm for pkm in ally.party if pkm.hp <= 0]))
        score += TYPE_CHART_MULTIPLIER[ally.active.type][opp.active.type] * 10
        return score

    def _minimax_evaluation(self, game_state: GameState, action: int, depth: int, alpha: float, beta: float) -> float:
        """Esegue la valutazione Minimax con Alpha-Beta Pruning."""
        if depth == 0 or self._is_terminal(game_state):
            return self._evaluate_gamestate(game_state)

        sim_state = copy.deepcopy(game_state)
        self._apply_action(sim_state, action)

        if depth % 2 == 0:
            max_eval = float('-inf')
            for opp_action in range(self.n_actions):
                score = self._minimax_evaluation(sim_state, opp_action, depth - 1, alpha, beta)
                max_eval = max(max_eval, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for opp_action in range(self.n_actions):
                score = self._minimax_evaluation(sim_state, opp_action, depth - 1, alpha, beta)
                min_eval = min(min_eval, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_eval

    def _calculate_dynamic_depth(self, game_state: GameState) -> int:
        """Determina dinamicamente la profondità del Minimax."""
        ally = game_state.teams[0]
        if len([pkm for pkm in ally.party if pkm.hp > 0]) < 2 or ally.active.hp < 30:
            return 2
        return 3

    def _is_terminal(self, game_state: GameState) -> bool:
        """Verifica se il gioco è terminato."""
        return all(pkm.hp <= 0 for pkm in game_state.teams[0].party + [game_state.teams[0].active]) or \
               all(pkm.hp <= 0 for pkm in game_state.teams[1].party + [game_state.teams[1].active])
