# Imposta il target predefinito
.DEFAULT_GOAL := run-all

# Obiettivi principali
.PHONY: run-minimax run-mcts run-mctsxminimax run-ux-minimax run-ux-mcts run-ux-mctsxminimax clean

# Variabile per l'agente (0=Minimax, 1=MCTS, 2=MCTSxMinimax)
minimax := 0
mcts := 1
mctsxminimax := 2

# Target per avviare solo main.py senza UX con un agente specifico
run-minimax:
	@cd .. && python3 gotta-battle-em-all/src/main.py $(minimax)

run-mcts:
	@cd .. && python3 gotta-battle-em-all/src/main.py $(mcts)

run-mctsxminimax:
	@cd .. && python3 gotta-battle-em-all/src/main.py $(mctsxminimax)

# Target per avviare sia PkmBattleUX.py (in background) che main.py con UX e un agente specifico
#Bisogna settare nel file main.py il flag ux a True
run-ux-minimax:
	@cd .. && python3 gotta-battle-em-all/ux/PkmBattleUX.py & \
	sleep 5 && \
	python3 src/main.py $(minimax)

run-ux-mcts:
	@cd .. && python3 gotta-battle-em-all/ux/PkmBattleUX.py & \
	sleep 5 && \
	python3 src/main.py $(mcts)

run-ux-mctsxminimax:
	@cd .. && python3 gotta-battle-em-all/ux/PkmBattleUX.py & \
	sleep 5 && \
	python3 src/main.py $(mctsxminimax)

# Target per pulire i file di log
clean:
	@rm -rf ../gotta-battle-em-all/logs/*
	@echo "Log files removed."
