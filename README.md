# Gotta-battle-em-all

This project contains a simulation environment for Pokémon battles, designed to test various strategies and agents. Below is an overview of the folder structure and instructions on how to run tests using the provided Makefile.

## Folder Structure

- **src/**: Contains four files:
  - `main.py`: The main simulation script that initializes and runs the battles.
  - `MCTS.py`: Contains the implementation of the Monte Carlo Tree Search (MCTS) agent.
  - `MCTSxMinimax.py`: Implements the MCTSxMinimax agent, combining MCTS and Minimax strategies.
  - `Minimax.py`: Contains the Minimax agent implementation.
- **logs/**: Stores log files generated during simulations and batch tests. During each test, a dedicated subfolder is created that contains logs for individual battles, as well as an additional file summarizing final statistics.
- **Makefile**: Provides commands to run simulations and tests, and to clean up logs.
- **ux/**: (Optional) Includes UX components for visualizing battles.

## Running Tests

To perform different actions, you can use the Makefile by executing `make <target>` in the root directory. Available targets include:

- **`run-minimax`**: Runs the simulation script using Minimax without the UX.
  ```
  make run-minimax
  ```

- **`run-ux-minimax`**: Launches both the UX (in the background) and the simulation script using Minimax.
  ```
  make run-ux-minimax
  ```

- **`run-tests-minimax`**: Performs a batch test using Minimax, running the simulation multiple times and generating aggregated statistics.
  ```
  make run-tests-minimax
  ```

- **`run-mcts`**: Runs the simulation script using Monte Carlo Tree Search (MCTS) without the UX.
  ```
  make run-mcts
  ```

- **`run-tests-mcts`**: Performs a batch test using MCTS.
  ```
  make run-tests-mcts
  ```

- **`run-ux-mcts`**: Launches both the UX (in the background) and the simulation script using MCTS.
  ```
  make run-ux-mcts
  ```

- **`run-mctsxminimax`**: Runs the simulation script using MCTSxMinimax without the UX.
  ```
  make run-mctsxminimax
  ```

- **`run-tests-mctsxminimax`**: Performs a batch test using MCTSxMinimax.
  ```
  make run-tests-mctsxminimax
  ```

- **`run-ux-mctsxminimax`**: Launches both the UX (in the background) and the simulation script using MCTSxMinimax.
  ```
  make run-ux-mctsxminimax
  ```

- **`clean`**: Removes all logs.
  ```
  make clean
  ```
