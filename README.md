# manusim

A versatile manufacturing simulation based on discrete event simulation with SimPy.

## Overview

manusim is a Python-based discrete-event simulation framework designed for modeling and analyzing manufacturing systems. It provides a flexible and customizable environment for simulating various aspects of a factory, including resource management, production scheduling, and product delivery.

## Key Features

-   **Discrete-Event Simulation:** Utilizes SimPy for accurate and efficient simulation of manufacturing processes.
-   **Modular Design:** Highly customizable and extensible architecture, allowing users to define their own simulation components and logic.
-   **Hydra Configuration:** Uses Hydra for managing simulation parameters, making it easy to configure and run experiments.
-   **Logging and Metrics:** Provides comprehensive logging and metrics collection for analyzing simulation results.

## Project Structure

The project is organized as follows:

-   `data/`: Contains experiment data and results.
-   `example/`: Contains an example simulation configuration and script.
    -   `conf/`: Contains the configuration files for the example simulation.
-   `notebooks/`: Contains Jupyter notebooks for data analysis and visualization.
-   `src/manusim/`: Contains the main source code for the simulation engine.
    -   `config/`: Contains default configuration files for Hydra.
    -   `engine/`: Contains the core simulation engine components.
    -   `experiment.py`: Defines the `ExperimentRunner` class for running multiple simulations.
    -   `factory_sim.py`: Defines the main `FactorySimulation` class.
    -   `metrics.py`: Contains functions for calculating and reporting metrics.

## Getting Started

### Prerequisites

-   Python 3.11 or higher
-   pip package manager

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/kleindaniell/manufacturing-simulation.git
    cd manufacturing-simulation
    ```

2.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running a Simulation

1.  Configure the simulation parameters in the `example/conf/` directory. The main configuration file is `config.yaml`.
2.  Run the example simulation using the following command:

    ```bash
    python example/example_sim.py
    ```

    This will start the simulation with the parameters defined in the `example/conf/` directory.

## Customization

To customize the simulation, you can modify the configuration files in the `example/conf/` directory. You can also extend the simulation by creating your own simulation class that inherits from `FactorySimulation` and implementing your own logic, as shown in `example/example_sim.py`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
