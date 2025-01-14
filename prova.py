import logging
from vgc.datatypes.Objects import PkmTeam
from vgc.behaviour.BattlePolicies import RandomPlayer, GUIPlayer, Minimax, PrunedBFS
from vgc.engine.PkmBattleEnv import PkmBattleEnv
from vgc.util.generator.PkmTeamGenerators import RandomTeamGenerator
from RandomAgent import Agent 
from multiprocessing.connection import Client
import time

# Configurazione logging
logging.basicConfig(
    filename="battle_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Generazione dei team
logging.info("Generazione dei team in corso...")
gen = RandomTeamGenerator()
team0 = gen.get_team().get_battle_team([0, 1, 2])
team1 = gen.get_team().get_battle_team([0, 1, 2])
agent0, agent1 = Agent(), RandomPlayer()

logging.info(f"Team0: {team0}")
logging.info(f"Team1: {team1}")

logging.info("Creazione connessione al server...")
address = ('localhost', 6000)
conn = Client(address, authkey='VGC AI'.encode('utf-8'))

# Inizializzazione dell'ambiente
logging.info("Inizializzazione dell'ambiente...")
env = PkmBattleEnv((team0, team1),
                   encode=(agent0.requires_encode(), agent1.requires_encode()), debug=True, conn=conn)

n_battles = 100  # numero di battaglie da eseguire
agent0_wins = 0  # Conta le vittorie di agent0

logging.info(f"Inizio delle {n_battles} battaglie...")

for battle in range(n_battles):
    logging.info(f"Inizio battaglia {battle + 1}...")
    t = False
    s, _ = env.reset()
    env.render(mode='ux')
    
    while not t:  # Continua finché la battaglia non termina
        start_time = time.time()
        a = [agent0.get_action(s[0]), agent1.get_action(s[1])]
        logging.info(f"Azioni pre-step: agent0={a[0]}, agent1={a[1]}")

        # Verifica che non siano None
        if a[0] is None or a[1] is None:
            logging.error(f"Azioni non valide: agent0={a[0]}, agent1={a[1]}")
            continue  # Salta questo ciclo e prova la prossima azione

        end_time = time.time()

        logging.info(f"Azioni prese in {end_time - start_time} secondi")
        print(f"Azioni prese in {end_time - start_time} secondi")
        logging.info(f"Mossa agent0: {a[0]}, Mossa agent1: {a[1]}")

        s, _, t, _, _ = env.step(a)
        logging.info(f"Stato aggiornato: {s}")
        env.render(mode='ux')
    
    # Dopo ogni partita, controlla il vincitore
    if env.winner == 0:  # Se agent0 vince
        agent0_wins += 1

    logging.info(f"Battaglia {battle + 1} conclusa. Vincitore: {'Agent0' if env.winner == 0 else 'Agent1'}")
    print(f"Battaglia {battle + 1} conclusa. Vincitore: {'Agent0' if env.winner == 0 else 'Agent1'}")
    logging.info(f"Stato finale della battaglia: {s}")
    print(f"Vittorie di agent0: {agent0_wins}/{battle + 1}")

# Calcola la percentuale di vittorie per agent0
win_percentage = (agent0_wins / n_battles) * 100
logging.info(f"Percentuale di vittorie di agent0: {win_percentage}%")
print(f"Percentuale di vittorie di agent0: {win_percentage}%")
