# from vgc.datatypes.Objects import PkmTeam
# from vgc.behaviour.BattlePolicies import RandomPlayer, GUIPlayer
# from vgc.engine.PkmBattleEnv import PkmBattleEnv
# from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator
# from RandomAgent import RandomAgent 
# from multiprocessing.connection import Client

# import time

# gen = RandomTeamGenerator()
# team0 = gen.get_team().get_battle_team([0, 1, 2])
# team1 = gen.get_team().get_battle_team([0, 1, 2])
# agent0, agent1 = RandomAgent(), RandomPlayer()
# address = ('localhost', 6000)

# conn = Client(address, authkey='VGC AI'.encode('utf-8'))


# env = PkmBattleEnv((team0, team1),
#                    encode=(agent0.requires_encode(), agent1.requires_encode()), debug=True, conn=conn)  # set new environment with teams
# n_battles = 3  # total number of battles
# t = False
# battle = 0
# while battle < n_battles:
#     s, _ = env.reset()
#     env.render(mode='ux')
#     while not t:  # True when all pkms of one of the two PkmTeam faint
#         start_time = time.time()
#         a = [agent0.get_action(s[0]), agent1.get_action(s[1])]
#         end_time = time.time()

#         print(f"Tempo impiegato: {end_time - start_time} secondi")
#         s, _, t, _, _ = env.step(a)  # for inference, we don't need reward
#         env.render(mode='ux')
#     t = False
#     battle += 1
# print(env.winner)  # winner id number



from vgc.datatypes.Objects import PkmTeam
from vgc.behaviour.BattlePolicies import RandomPlayer, GUIPlayer
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator
from RandomAgent import Agent 
from multiprocessing.connection import Client
import time

# Generazione dei team
gen = RandomTeamGenerator()
team0 = gen.get_team().get_battle_team([0, 1, 2])
team1 = gen.get_team().get_battle_team([0, 1, 2])
agent0, agent1 = Agent(), RandomPlayer()

address = ('localhost', 6000)
conn = Client(address, authkey='VGC AI'.encode('utf-8'))

# Inizializzazione dell'ambiente
env = PkmBattleEnv((team0, team1),
                   encode=(agent0.requires_encode(), agent1.requires_encode()), debug=True, conn=conn)

n_battles = 30  # numero di battaglie da eseguire
agent0_wins = 0  # Conta le vittorie di agent0

for battle in range(n_battles):
    t = False
    s, _ = env.reset()
    env.render(mode='ux')
    
    while not t:  # Continua finché la battaglia non termina
        start_time = time.time()
        a = [agent0.get_action(s[0]), agent1.get_action(s[1])]
        end_time = time.time()

        print(f"Tempo impiegato: {end_time - start_time} secondi")
        s, _, t, _, _ = env.step(a)
        env.render(mode='ux')
    
    # Dopo ogni partita, controlla il vincitore
    if env.winner == 0:  # Se agent0 vince
        agent0_wins += 1
    
    print(f"Vittorie di agent0: {agent0_wins}/{battle + 1}")

# Calcola la percentuale di vittorie per agent0
win_percentage = (agent0_wins / n_battles) * 100
print(f"Percentuale di vittorie di agent0: {win_percentage}%")
