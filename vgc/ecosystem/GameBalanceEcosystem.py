from copy import deepcopy
from typing import List

from vgc.balance.meta import StandardMetaData, MetaEvaluator
from vgc.balance.restriction import VGCDesignConstraints
from vgc.competition.Competition import Competitor
from vgc.competition.Competitor import CompetitorManager
from vgc.datatypes.Constants import DEFAULT_MATCH_N_BATTLES
from vgc.datatypes.Objects import PkmRoster
from vgc.ecosystem.BattleEcosystem import Strategy
from vgc.ecosystem.ChampionshipEcosystem import ChampionshipEcosystem


class GameBalanceEcosystem:

    def __init__(self, evaluator: MetaEvaluator, competitor: Competitor, surrogate_agent: List[CompetitorManager],
                 constraints: VGCDesignConstraints, base_roster: PkmRoster, meta_data: StandardMetaData, debug=False,
                 render=False, n_battles=DEFAULT_MATCH_N_BATTLES, strategy: Strategy = Strategy.RANDOM_PAIRING):
        self.evaluator = evaluator
        self.c = competitor
        self.constraints = constraints
        self.meta_data = meta_data
        self.base_roster = deepcopy(base_roster)
        self.total_score = 0.0
        self.vgc: ChampionshipEcosystem = ChampionshipEcosystem(base_roster, meta_data, debug, render, n_battles,
                                                                strategy=strategy)
        for c in surrogate_agent:
            self.vgc.register(c)

    def run(self, n_epochs, n_vgc_epochs: int, n_league_epochs: int):
        for epoch in range(n_epochs):
            self.vgc.run(n_vgc_epochs, n_league_epochs)
            if epoch > 0:
                self.total_score += self.evaluator.eval(self.meta_data, self.base_roster)
            delta_roster = self.c.balance_policy.get_action((deepcopy(self.vgc.roster), deepcopy(self.meta_data),
                                                             self.constraints))
            copy_roster = deepcopy(self.vgc.roster)
            delta_roster.apply(copy_roster)
            violated_rules = self.constraints.check_every_rule(copy_roster)
            if len(violated_rules) == 0:
                self.meta_data.update_with_delta_roster(delta_roster)
                self.vgc.roster_ver += 1
