from vgc.datatypes.Objects import PkmTeam
from vgc.behaviour.BattlePolicies import RandomPlayer
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator
from Agents import Minimax_Agent, MinimaxNodes_Agent
from multiprocessing.connection import Client
import time

gen = RandomTeamGenerator()
team0 = gen.get_team().get_battle_team([0, 1, 2])
team1 = gen.get_team().get_battle_team([0, 1, 2])
agent0, agent1 = Minimax_Agent(), RandomPlayer()
address = ('localhost', 6000)

conn = Client(address, authkey='VGC AI'.encode('utf-8'))


env = PkmBattleEnv((team0, team1),
                   encode=(agent0.requires_encode(), agent1.requires_encode()), debug=True, conn=conn)  # set new environment with teams

n_battles = 3  # total number of battles
t = False
battle = 0
wins = [0,0]
while battle < n_battles:
    s, _ = env.reset()
    env.render(mode='ux')
    while not t:  # True when all pkms of one of the two PkmTeam faint
        start_time = time.time()
        a = [agent0.get_action(s[0]), agent1.get_action(s[1])]
        s, _, t, _, _ = env.step(a)  # for inference, we don't need reward
        end_time = time.time()
        print(f"Step time: {end_time - start_time} seconds")
        env.render(mode='ux')
    print(f"Game {battle+1} ended")
    wins[env.winner] += 1
    battle += 1
    if wins[0] < 2 and wins[1] < 2 :
        t = False
    else: break


if(wins[0] > wins[1]):
    print("Abbiamo vinto!")
else: print("Abbiamo perso")  # winner id number

# gen = RandomTeamGenerator()
# full_team0 = gen.get_team()
# full_team1 = gen.get_team()
# conn = Client(address, authkey='VGC AI'.encode('utf-8'))
# env = PkmBattleEnv((full_team0.get_battle_team([0, 1, 2]), full_team1.get_battle_team([0, 1, 2])), debug=True,
#                    conn=conn, encode=(a0.requires_encode(), a1.requires_encode()))
# env.reset()
# t = False
# ep = 0
# n_battles = 3
# while ep < n_battles:
#     s, _ = env.reset()
#     env.render(mode='ux')
#     ep += 1
#     while not t:
#         a = [a0.get_action(s[0]), a1.get_action(s[1])]
#         s, _, t, _, _ = env.step(a)
#         env.render(mode='ux')
#     t = False
# env.close()
