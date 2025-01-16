# Imposta il target predefinito
.DEFAULT_GOAL := run-all

# Obiettivi principali
.PHONY: run-prova run-all run-tests clean

# Numero di battaglie (modificabile)
NUM_BATTLES := 5
BATCH_RUNS := 10

# Target per avviare solo provagiulio.py senza UX
run-prova:
	@cd .. && python3 gotta-battle-em-all/src/provagiulio.py --n-battles $(NUM_BATTLES)

# Target per avviare sia PkmBattleUX.py (in background) che provagiulio.py con UX
run-all:
	@cd .. && python3 gotta-battle-em-all/ux/PkmBattleUX.py & \
	sleep 5 && \
	python3 src/provagiulio.py --use-ux --n-battles $(NUM_BATTLES)

# Target per eseguire un batch di test
run-tests:
	@cd .. && python3 gotta-battle-em-all/src/provagiulio.py --n-battles $(NUM_BATTLES) --test-batch --batch-runs $(BATCH_RUNS)

# Target per pulire i file di log
clean:
	@rm -rf ../gotta-battle-em-all/logs/*
	@echo "Log files removed."
