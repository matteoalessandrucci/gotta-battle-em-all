from vgc.datatypes.Objects import PkmTeam
from vgc.behaviour.BattlePolicies import RandomPlayer, GUIPlayer
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator
from RandomAgent import RandomAgent
from multiprocessing.connection import Client

gen = RandomTeamGenerator()
team0 = gen.get_team().get_battle_team([0, 1, 2])
team1 = gen.get_team().get_battle_team([0, 1, 2])
agent0, agent1 = RandomAgent(), RandomPlayer()
address = ('localhost', 6000)

conn = Client(address, authkey='VGC AI'.encode('utf-8'))


env = PkmBattleEnv((team0, team1),
                   encode=(agent0.requires_encode(), agent1.requires_encode()), debug=True, conn=conn)  # set new environment with teams
n_battles = 3  # total number of battles
t = False
battle = 0
while battle < n_battles:
    s, _ = env.reset()
    env.render(mode='ux')
    while not t:  # True when all pkms of one of the two PkmTeam faint
        a = [agent0.get_action(s[0]), agent1.get_action(s[1])]
        s, _, t, _, _ = env.step(a)  # for inference, we don't need reward
        env.render(mode='ux')
    t = False
    battle += 1
print(env.winner)  # winner id number

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
