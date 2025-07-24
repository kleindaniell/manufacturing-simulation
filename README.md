# manusim

A versatile manufacturing simulation based on discrete event simulation with SimPy.

## Overview

manusim is a Python-based discrete-event simulation framework designed for modeling and analyzing manufacturing systems. It provides a flexible and customizable environment for simulating various aspects of a factory, including resource management, production scheduling, and product delivery.

## Key Features

-   **Discrete-Event Simulation:** Utilizes SimPy for accurate and efficient simulation of manufacturing processes.
-   **Modular Design:** Highly customizable and extensible architecture, allowing users to define their own simulation components and logic.
-   **Hydra Configuration:** Uses Hydra for managing simulation parameters, making it easy to configure and run experiments.
-   **Logging and Metrics:** Provides comprehensive logging and metrics collection for analyzing simulation results.
-   **Experiment Runner:** Includes a powerful tool for running multiple simulation scenarios and aggregating the results.

## Core Concepts and Functionality

To understand how `manusim` works, it's helpful to think of it as a digital twin of a real-world factory (e.g., a car factory). The simulation is built around a few core concepts that model the key operations of a manufacturing plant.

### The Simulation Environment and Configuration

The simulation is orchestrated by the `FactorySimulation` class, which acts as the main blueprint for a manufacturing facility. When you initialize it, you provide three key configurations:

-   **`config`**: Defines global rules like the simulation duration (`run_until`), warm-up time, and product delivery policies.
-   **`resources`**: Describes every machine and workstation (e.g., stamping press, welding robots, paint booth), including how many are available and their specific capabilities.
-   **`products`**: Details each product the factory can make (e.g., different car models) and the sequence of operations required to build them.

### The Warm-up Period

A simulation doesn't start collecting performance data from the very first second. The `warmup` period allows the simulation to run for a specified time to reach a realistic, steady state. This ensures that the metrics are not skewed by the initial, unnaturally empty state of the factory. Think of it as letting an assembly line get fully up and running before measuring its output.

### Randomness and Distributions

Real-world factories have variability. `manusim` models this using a `DistributionGenerator` for key processes:
-   **Product Demand**: Simulates the random arrival of new customer orders.
-   **Processing Times**: Models the slight variations in the time it takes a machine to complete a task.
-   **Breakdowns**: Simulates the time between machine failures (`TBF`) and the time it takes to repair them (`TTR`), making the simulation more realistic.

### Core Factory Processes

The simulation runs as a series of interconnected processes that mimic factory life:

1.  **Production System (`_production_system`)**: This is the heart of the factory's operation for each resource. In a continuous loop, it checks for potential breakdowns, selects the next production order from its queue, handles any necessary setup time (e.g., changing a tool or paint color), processes the order, and finally sends the completed part to the next stage.

2.  **Transportation (`_transportation`)**: This process models the material handling system, such as conveyor belts or automated guided vehicles (AGVs). It takes a finished part from one workstation's output and moves it to the input queue of the next workstation in the sequence. If the part has completed its final processing step, it is moved to the finished goods inventory.

3.  **Breakdowns (`_breakdowns`)**: This function simulates machine failures. When a machine's operational time exceeds its "time between failure" threshold, this process is triggered, making the resource unavailable for a specified "time to repair." This models the real-world impact of maintenance and equipment failure on production.

4.  **Demand, Scheduling, and Delivery**:
    -   **`_generate_demand_orders`**: Simulates the arrival of new customer orders with random frequency, quantity, and due dates.
    -   **`scheduler`**: Acts as the production planner, converting customer `DemandOrders` into internal `ProductionOrders` that are released to the factory floor.
    -   **`_delivery_orders`**: Manages the shipment of finished goods according to predefined policies like shipping as soon as the product is ready (`asReady`) or waiting until the specific due date (`onDue`).

## Engine Architecture

The core logic of the simulation is encapsulated in the `src/manusim/engine/` directory. Understanding these components is key to customizing and extending the framework.

-   **`orders.py`**: Defines the data structures for orders.
    -   `DemandOrder`: Represents an external request from a customer, containing details like product, quantity, and due date.
    -   `ProductionOrder`: Represents an internal work order that is processed on the factory floor. It's created by the `scheduler` based on a `DemandOrder`.

-   **`stores.py`**: Manages the state of the entire simulation using `simpy.Store` and `simpy.FilterStore`. It holds all the dynamic elements of the factory:
    -   Input queues for each resource.
    -   Work-in-Progress (WIP) inventory.
    -   Finished Goods (FG) inventory.
    -   Transportation channels between resources.

-   **`logs.py`**: Contains classes for collecting time-stamped data during the simulation.
    -   `ProductLogs`: Tracks product-specific metrics like flow time, tardiness, and finished goods levels.
    -   `ResourceLogs`: Tracks resource-specific metrics like utilization, setup times, and breakdowns.
    -   `GeneralLogs`: A flexible container for any other custom logs you wish to add.

-   **`utils.py`**: Provides helper functions and classes.
    -   `DistributionGenerator`: A critical utility that generates random numbers from various statistical distributions (e.g., constant, uniform, normal). This is used to inject realistic variability into processing times, demand arrivals, and machine breakdowns.

## Project Structure

The project is organized as follows:

-   `data/`: Contains experiment data and results.
-   `examples/`: Contains example simulation configurations and scripts.
    -   `conf/`: Contains the configuration files for the example simulation.
-   `notebooks/`: Contains Jupyter notebooks for data analysis and visualization.
-   `src/manusim/`: Contains the main source code for the simulation engine.
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

1.  Configure the simulation parameters in the `examples/simple_scheduler/conf/` directory. The main configuration file is `config.yaml`.
2.  Run the example simulation using the following command:

    ```bash
    python examples/simple_scheduler/simulation.py
    ```

    This will start the simulation with the parameters defined in the `examples/simple_scheduler/conf/` directory.

### Running Experiments

For more advanced use cases, the `ExperimentRunner` class allows you to run multiple simulations with different seeds, making it ideal for statistical analysis.

To run an experiment, you can create a script that initializes the `FactorySimulation` and passes it to the `ExperimentRunner`. The runner will handle the execution of multiple runs and save the results in a structured format.

```python
# Example of running an experiment
from manusim.experiment import ExperimentRunner
from examples.simple_scheduler.simulation import NewSimulation

# Initialize the simulation
sim = NewSimulation(config, resources, products)

# Run the experiment
runner = ExperimentRunner(
    simulation=sim,
    number_of_runs=10,
    save_folder_path="data/experiments",
    run_name="my_experiment"
)
runner.run_experiment()
```

## Customization

To customize the simulation, you can modify the configuration files or extend the `FactorySimulation` class to implement your own logic. This is the recommended way to create a custom simulation.

The `FactorySimulation` class is an abstract base class that can be extended to create a custom simulation. The following methods can be overridden to implement custom logic:

-   `__init__(self, ...)`: The constructor can be extended to initialize custom attributes and logic.
-   `order_selection(self, resource)`: This method can be overridden to define how the next production order is selected from a resource's queue.
-   `scheduler(self)`: This method can be overridden to implement a custom scheduling logic.
-   `_start_custom_process(self)`: This method can be used to start custom simulation processes.
-   `_create_custom_logs(self)`: This method can be implemented to create custom logs for tracking specific metrics.
-   `_register_custom_logs(self)`: This method can be used to register the custom logs.
-   `print_custom_metrics(self)`: This method can be implemented to print custom metrics.
-   `process_fg_reduce(self, product)`: This method can be overridden to update custom logic when finished goods are reduced.

### Example 1: Simple Scheduler

The `simple_scheduler` example demonstrates how to implement a custom scheduling logic by overriding the `order_selection` method. In this example, the scheduler selects the order with the highest priority from the resource's queue.

```python
class NewSimulation(FactorySimulation):
    def __init__(
        self,
        config: dict,
        resources: dict,
        products: dict,
        print_mode="all",
        seed: int = None,
    ):
        super().__init__(
            config,
            resources,
            products,
            print_mode,
            seed,
        )

    # New order selection method
    def order_selection(self, resource):
        orders: List[ProductionOrder] = self.stores.resource_input[resource].items

        # Return order with highest priority
        if len(orders) > 0:
            selected_order = max(orders, key=lambda x: x.priority)
            # Get order from queue
            productionOrder = yield self.stores.resource_input[resource].get(
                lambda x: x.id == selected_order.id
            )
        else:
            productionOrder = yield self.stores.resource_input[resource].get()

        return productionOrder
```

### Example 2: Drum-Buffer-Rope (DBR)

The `toc_dbr` example implements the Drum-Buffer-Rope (DBR) scheduling methodology from the Theory of Constraints (TOC). This example showcases a more advanced customization by overriding multiple methods to control the release of orders and manage buffers.

```python
class DBRSimulation(FactorySimulation):
    def __init__(
        self,
        config: dict,
        resources: dict,
        products: dict,
        print_mode="all",
        seed: int = None,
    ):
        super().__init__(
            config,
            resources,
            products,
            print_mode,
            seed,
        )

        self.order_release_limit = self.config.get("order_release_limit", float("inf"))
        self.ccr_release_limit = self.config.get("ccr_release_limit", False)
        self.scheduler_interval = self.config.get("scheduler_interval", 72)
        self.cb_target_level = self.config.get("cb_target_level", float("inf"))

        self._create_constraint_buffer()
        self._create_shipping_buffers()

    def _start_custom_process(self):
        self.contraint_resource, self.utilization_df = self.define_constraint()
        self.env.process(self._update_constraint_buffer(self.contraint_resource))
        self.env.process(self._process_demandOrders())
        # Update finihed goods
        self.sb_update: bool = self.config.get("sb_update", False)
        self.cb_update: bool = self.config.get("cb_update", False)
        for product in self.products_config.keys():
            self.env.process(self._update_finished_goods(product))
            if self.sb_update:
                self.env.process(self.adjust_shipping_buffer(product))
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
