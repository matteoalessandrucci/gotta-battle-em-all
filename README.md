# Gotta-battle-em-all

This project contains a simulation environment for Pokémon battles, designed to test various strategies and agents. Below is an overview of the folder structure and instructions on how to run tests using the provided Makefile.

## Folder Structure

- **src/**: Contains the main simulation script and related classes.
- **logs/**: Stores log files generated during simulations and batch tests.
- **Makefile**: Provides commands to run simulations and tests, and to clean up logs.
- **ux/**: (Optional) Includes UX components for visualizing battles.

## Running Tests

To perform different actions, you can use the Makefile by executing `make <target>` in the root directory. Available targets include:

- **`run-prova`**: Runs the simulation script without the UX.
  ```
  make run-prova
  ```

- **`run-all`**: Launches both the UX (in the background) and the simulation script.
  ```
  make run-all
  ```

- **`run-tests`**: Performs a batch test, running the simulation multiple times and generating aggregated statistics.
  ```
  make run-tests
  ```

- **`clean`**: Removes all logs.
  ```
  make clean
  ```

