from vgc.datatypes.Objects import PkmTeam
from vgc.behaviour.BattlePolicies import RandomPlayer, GUIPlayer, Minimax
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator
from Agents import RandomAgent, MonteCarloAgent, MonteCarloMinimaxAgent, MinimaxWithAlphaBeta
from multiprocessing.connection import Client

# OK
#     gen = RandomTeamGenerator()
#     team0 = gen.get_team().get_battle_team([0, 1, 2])
#     team1 = gen.get_team().get_battle_team([0, 1, 2])
#     agent0, agent1 = MonteCarloMinimaxAgent(simulations=5), MinimaxWithAlphaBeta()
#     address = ('localhost', 6000)

#     conn = Client(address, authkey='VGC AI'.encode('utf-8'))


#     env = PkmBattleEnv((team0, team1),
#                     encode=(agent0.requires_encode(), agent1.requires_encode()), debug=True, conn=conn)  # set new environment with teams
#     n_battles = 3  # total number of battles
#     t = False
#     battle = 0
#     while battle < n_battles:
#         s, _ = env.reset()
#         env.render(mode='ux')
#         while not t:  # True when all pkms of one of the two PkmTeam faint
#             a = [agent0.get_action(s[0]), agent1.get_action(s[1])]
#             s, _, t, _, _ = env.step(a)  # for inference, we don't need reward
#             env.render(mode='ux')
#         t = False
#         battle += 1
#     if env.winner: print("Vittoria!")  # winner id number
#     else: print("Sconfitta :c")


#GIA COMMENTATO
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
# Genera una squadra randomica

gen = RandomTeamGenerator()
team0 = gen.get_team().get_battle_team([0, 1, 2])
team1 = gen.get_team().get_battle_team([0, 1, 2])

# Crea gli agenti
agent0, agent1 = RandomAgent(), MonteCarloMinimaxAgent()

# Definisci l'indirizzo di connessione
address = ('localhost', 6000)
conn = Client(address, authkey='VGC AI'.encode('utf-8'))

# Crea l'ambiente di battaglia con le squadre
env = PkmBattleEnv((team0, team1),
                   encode=(agent0.requires_encode(), agent1.requires_encode()), 
                   debug=True, conn=conn)

n_battles = 100
vittorie=0

# Inizia le battaglie
battle = 0
while battle < n_battles:
    s, _ = env.reset()  # resetta l'ambiente prima di ogni battaglia
    env.render(mode='ux')  # renderizza lo stato iniziale della battaglia
    t = False  # flag che indica se la battaglia è finita
    
    while not t:  # continua finché una delle squadre non perde tutti i Pokémon
        a = [agent0.get_action(s[0]), agent1.get_action(s[1])]  # ottieni le azioni per entrambe le squadre
        s, _, t, _, _ = env.step(a)  # esegui un passo nell'ambiente
        env.render(mode='ux')  # renderizza l'andamento della battaglia
    
    # Dopo la battaglia, memorizza il risultato
    if env.winner:
        print(f"Battaglia {battle + 1}: Vittoria! (Squadra {env.winner})")
        vittorie+=1
    else:
        print(f"Battaglia {battle + 1}: Sconfitta :c")
    
    battle += 1  # passa alla prossima battaglia

conn.close()
percentuale_vittorie = (vittorie / n_battles) * 100
print(f"La percentuale delle vittorie è: {percentuale_vittorie}")

