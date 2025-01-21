# Gotta-battle-em-all

This project provides a simulation environment for Pokémon battles, designed to test and evaluate various strategies and agents.

## Folder Structure

- **src/**: Contains the main simulation script and agent implementations:
  - `main.py`: Main script that initializes and runs the battles. Agents are selected using a numeric argument: `0` for Minimax, `1` for MCTS, and `2` for MCTSxMinimax.
  - `MCTS.py`: Implements the Monte Carlo Tree Search (MCTS) agent.
  - `MCTSxMinimax.py`: Combines MCTS and Minimax strategies.
  - `Minimax.py`: Implements the Minimax agent.
- **logs/**: Stores logs for each battle or batch test. Logs are organized by agent type (`minimax`, `mcts`, or `mcts_x_minimax`).
- **Makefile**: Provides commands to execute simulations and clean logs.
- **ux/**: (Optional) Contains UX components for visualizing battles.

## Setting Parameters in `main.py`

Some parameters are hardcoded in the script. Adjust them as necessary:

1. **Agent Selection**: 
   - Pass an argument to `main.py` to select the agent:
     - `0`: Minimax
     - `1`: MCTS
     - `2`: MCTSxMinimax

   Example:
   ```bash
   python3 src/main.py 1  # Runs with MCTS agent
   ```

2. **Adjustable Parameters** (inside `main.py`):
   - **`n_battles`**: Number of battles to simulate (default: 5).
   - **`use_ux`**: Set to `True` to enable UX visualization.
   - **`test_batch`**: Set to `True` to enable batch testing.
   - **`batch_runs`**: Number of batch runs if `test_batch` is enabled (default: 10).

## Using the Makefile

The Makefile provides shortcuts for executing various simulations. Run these commands in the root directory:

### Running Simulations Without UX

- **`run-minimax`**: Runs the simulation with the Minimax agent.
  ```bash
  make run-minimax
  ```

- **`run-mcts`**: Runs the simulation with the MCTS agent.
  ```bash
  make run-mcts
  ```

- **`run-mctsxminimax`**: Runs the simulation with the MCTSxMinimax agent.
  ```bash
  make run-mctsxminimax
  ```

### Running Simulations With UX

To visualize battles, use these commands. Ensure the `use_ux` parameter is set to `True` in `main.py`:

- **`run-ux-minimax`**: Launches the UX and runs the simulation with the Minimax agent.
  ```bash
  make run-ux-minimax
  ```

- **`run-ux-mcts`**: Launches the UX and runs the simulation with the MCTS agent.
  ```bash
  make run-ux-mcts
  ```

- **`run-ux-mctsxminimax`**: Launches the UX and runs the simulation with the MCTSxMinimax agent.
  ```bash
  make run-ux-mctsxminimax
  ```

### Cleaning Logs

- **`clean`**: Removes all log files.
  ```bash
  make clean
  ```
