import os
import sys
import time
from math import ceil
from datetime import datetime
from multiprocessing.connection import Client
from Minimax import Minimax_Agent
from MCTSxMinimax import MCTSxMinimax
from MCTS import MonteCarloAgent
sys.path.append(os.path.join(sys.path[0], '..'))

from vgc.datatypes.Objects import PkmTeam
from vgc.behaviour.BattlePolicies import RandomPlayer
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator

class BattleSimulator:
    def __init__(self, agent_type=0, use_ux=True, n_battles=3, batch_folder=None):
        """
        Initializes the battle simulator.
        - agent_type: Numeric argument (0=Minimax, 1=MCTS, 2=MCTSxMinimax).
        - use_ux: If True, connects to the UX for rendering battles.
        - n_battles: Total number of battles to simulate.
        - batch_folder: Folder path to save logs if part of a batch test.
        """
        self.use_ux = use_ux
        self.n_battles = n_battles
        self.batch_folder = batch_folder
        self.agent_type = agent_type
        self.log = []
        self.total_turns = 0
        self.wins = [0, 0]
        self.win_threshold = ceil(self.n_battles / 2)
        self.team0, self.team1 = self.initialize_teams()

        # Initialize the agent based on the agent_type
        if self.agent_type == 1:
            self.agent0 = MonteCarloAgent()
        elif self.agent_type == 2:
            self.agent0 = MCTSxMinimax()
        else:
            self.agent0 = Minimax_Agent()
        
        self.agent1 = RandomPlayer()
        self.env = self.initialize_environment()

    def initialize_teams(self):
        """Generates two random teams for the battle."""
        gen = RandomTeamGenerator()
        return gen.get_team().get_battle_team([0, 1, 2]), gen.get_team().get_battle_team([0, 1, 2])

    def initialize_environment(self):
        """
        Initializes the battle environment.
        Connects to the UX if enabled.
        """
        conn = Client(('localhost', 6000), authkey='VGC AI'.encode('utf-8')) if self.use_ux else None
        return PkmBattleEnv(
            (self.team0, self.team1),
            encode=(self.agent0.requires_encode(), self.agent1.requires_encode()),
            debug=True,
            conn=conn
        )

    def log_to_file(self, log_text, filename=None):
        """
        Saves logs to a file in the specified directory.
        """
        agent_type_map = {0: "minimax", 1: "mcts", 2: "mcts_x_minimax"}
        agent_type = agent_type_map.get(self.agent_type, "unknown")

        base_dir = f"gotta-battle-em-all/logs/{agent_type}" if not self.use_ux else f"logs/{agent_type}"
        log_dir = os.path.join(base_dir, self.batch_folder) if self.batch_folder else base_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{'test_batch_' if self.batch_folder else 'battle_'}log_{agent_type}_{timestamp}.txt")

        print(f"Saving log to: {log_file}")
        with open(log_file, "w") as f:
            f.write(log_text)

    def run_battles(self):
        """
        Simulates a series of battles, logging results.
        """
        agent_type_map = {0: "Minimax", 1: "Monte Carlo Agent", 2: "MCTSxMinimax"}
        self.log.append(f"Using {agent_type_map.get(self.agent_type, 'Unknown')} Agent\n")
        self.log.append(f"Starting {self.n_battles} battles\n")
        print(self.log[-1].strip())

        for battle in range(self.n_battles):
            t = False
            s, _ = self.env.reset()
            if self.use_ux:
                self.env.render(mode='ux')

            battle_turns = 0

            while not t:
                start_time = time.time()
                a = [self.agent0.get_action(s[0]), self.agent1.get_action(s[1])]
                s, _, t, _, _ = self.env.step(a)
                step_time = time.time() - start_time

                message = f"Battle {battle + 1}, Turn {battle_turns + 1}: Step time = {step_time:.4f} seconds\n"
                self.log.append(message)
                print(message.strip())

                if self.use_ux:
                    self.env.render(mode='ux')

                battle_turns += 1

            message = f"Battle {battle + 1} ended in {battle_turns} turns\n"
            self.log.append(message)
            print(message.strip())

            self.wins[self.env.winner] += 1
            self.total_turns += battle_turns

            if self.wins[0] >= self.win_threshold or self.wins[1] >= self.win_threshold:
                break

        self.finalize_results()

    def finalize_results(self):
        """
        Finalizes the results, calculates win rates, and logs them.
        """
        total_battles = sum(self.wins)
        win_rate_team0 = (self.wins[0] / total_battles * 100) if total_battles else 0
        win_rate_team1 = (self.wins[1] / total_battles * 100) if total_battles else 0

        results = [
            "\nFinal Results:\n",
            f"Team 0 Wins: {self.wins[0]}\n",
            f"Team 1 Wins: {self.wins[1]}\n",
            f"Winner Team 0!\n" if self.wins[0] > self.wins[1] else "Winner Team 1!\n",
            f"Win Rate (Team 0): {win_rate_team0:.2f}%\n",
            f"Win Rate (Team 1): {win_rate_team1:.2f}%\n",
            f"Total Battles: {total_battles}\n",
            f"Total Turns: {self.total_turns}\n",
            f"Average Turns per Battle: {self.total_turns / total_battles:.2f}\n"
        ]

        for result in results:
            self.log.append(result)
            print(result.strip())

        self.log_to_file("".join(self.log))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Pokémon battles.")
    parser.add_argument("agent_type", type=int, choices=[0, 1, 2], help="Select agent type: 0 for Minimax, 1 for MCTS, 2 for MCTSxMinimax")
    args = parser.parse_args()

    # Static arguments
    n_battles = 5
    use_ux = False
    test_batch = False
    batch_runs = 10

    if test_batch:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        batch_folder = f"batch_test_{timestamp}"

        total_wins_team0 = 0
        total_wins_team1 = 0

        for _ in range(batch_runs):
            simulator = BattleSimulator(agent_type=args.agent_type, use_ux=use_ux, n_battles=n_battles, batch_folder=batch_folder)
            simulator.run_battles()

            if simulator.wins[0] > simulator.wins[1]:
                total_wins_team0 += 1
            else:
                total_wins_team1 += 1

        total_battles = batch_runs
        win_rate_team0 = total_wins_team0 / total_battles * 100
        win_rate_team1 = total_wins_team1 / total_battles * 100

        log = [
            f"Batch Results (over {batch_runs} runs):\n",
            f"Total Battles: {total_battles}\n",
            f"Total Wins Team 0: {total_wins_team0}\n",
            f"Total Wins Team 1: {total_wins_team1}\n",
            f"Win Rate (Team 0): {win_rate_team0:.2f}%\n",
            f"Win Rate (Team 1): {win_rate_team1:.2f}%\n"
        ]

        simulator.log_to_file("".join(log), filename="test_batch_log_stats")
        print("Batch testing completed. Log saved.")
    else:
        simulator = BattleSimulator(agent_type=args.agent_type, use_ux=use_ux, n_battles=n_battles)
        simulator.run_battles()
        print("Battles completed. Log saved.")
