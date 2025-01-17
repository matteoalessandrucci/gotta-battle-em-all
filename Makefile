# Imposta il target git edefinito
.DEFAULT_GOAL := run-all

# Obiettivi principali
.PHONY: run-minimax run-ux-minimax run-tests-minimax run-tests-mcts run-tests-mctsxminimax clean run-mcts run-ux-mcts run-mctsxminimax run-ux-mctsxminimax

# Numero di battaglie (modificabile)
NUM_BATTLES := 5
BATCH_RUNS := 10

# Target per avviare solo main.py senza UX
run-minimax:
	@cd .. && python3 gotta-battle-em-all/src/main.py --n-battles $(NUM_BATTLES)

# Target per avviare sia PkmBattleUX.py (in background) che main.py con UX
run-ux-minimax:
	@cd .. && python3 gotta-battle-em-all/ux/PkmBattleUX.py & \
	sleep 5 && \
	python3 src/main.py --use-ux --n-battles $(NUM_BATTLES)

# Target per eseguire un batch di test con Minimax
run-tests-minimax:
	@cd .. && python3 gotta-battle-em-all/src/main.py --n-battles $(NUM_BATTLES) --test-batch --batch-runs $(BATCH_RUNS)

# Target per avviare solo main.py con MCTS
run-mcts:
	@cd .. && python3 gotta-battle-em-all/src/main.py --n-battles $(NUM_BATTLES) --use-mcts

# Target per eseguire un batch di test con MCTS
run-tests-mcts:
	@cd .. && python3 gotta-battle-em-all/src/main.py --n-battles $(NUM_BATTLES) --test-batch --batch-runs $(BATCH_RUNS) --use-mcts

# Target per avviare sia PkmBattleUX.py che main.py con UX e MCTS
run-ux-mcts:
	@cd .. && python3 gotta-battle-em-all/ux/PkmBattleUX.py & \
	sleep 5 && \
	python3 src/main.py --use-ux --use-mcts --n-battles $(NUM_BATTLES)

# Target per avviare solo main.py con MCTSxMinimax
run-mctsxminimax:
	@cd .. && python3 gotta-battle-em-all/src/main.py --n-battles $(NUM_BATTLES) --use-mcts-minimax

# Target per eseguire un batch di test con MCTSxMinimax
run-tests-mctsxminimax:
	@cd .. && python3 gotta-battle-em-all/src/main.py --n-battles $(NUM_BATTLES) --test-batch --batch-runs $(BATCH_RUNS) --use-mcts-minimax

# Target per avviare sia PkmBattleUX.py che main.py con UX e MCTSxMinimax
run-ux-mctsxminimax:
	@cd .. && python3 gotta-battle-em-all/ux/PkmBattleUX.py & \
	sleep 5 && \
	python3 src/main.py --use-ux --use-mcts-minimax --n-battles $(NUM_BATTLES)

# Target per pulire i file di log
clean:
	@rm -rf ../gotta-battle-em-all/logs/*
	@echo "Log files removed."
