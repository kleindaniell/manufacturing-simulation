import simpy
import random
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from time import sleep
import timeit
from gymnasium import spaces
import gymnasium as gym


class Environment(gym.env):
    def __init__(
        self,
        run_until: int,
        resources_cfg: dict,
        products_cfg: dict,
        raw_material_cfg: dict,
        schedule_interval: int,
        buffer_adjust_interval: int,
        set_constraint: int = None,
        sim_monitor_interval: int = None,
        monitor_interval: int = None,
        warmup: bool = False,
        actions: str = "discret",
        env_mode: str = "simulation",
        seed: int = None,
    ):
        super().__init__()
        random.seed(seed)
        self.env = simpy.Environment()

        # Parameters
        self.resources_config = resources_cfg
        self.products_config = products_cfg
        self.raw_material = raw_material_cfg
        self.warmup = warmup
        self.run_until = run_until
        self.sim_monitor_interval = sim_monitor_interval
        self.monitor_interval = monitor_interval
        self.actions_type = actions
        self.env_mode = env_mode
        self.schedule_interval = schedule_interval
        self.buffer_adjust_interval = buffer_adjust_interval
        self.constraint = set_constraint

        self._create_resources()

        if self.sim_monitor_interval:
            self.env.process(self._simulation_monitor(self.sim_monitor_interval))
            self.env.process(self._monitor(interval=self.monitor_interval))

    def _process_production(self, process):

        resource = self.processes_config[process]["resource"]

        constraint = False
        if resource == self.constraint:
            constraint = True
        else:
            constraint = False

        final_process = False
        next_process = False
        for product in self.products_config:
            if self.products_config[product]["final_process"] == process:
                final_process = True
            else:
                next_process = self.processes_config[process]["next"]

        process_deps = self.processes_config[process]["deps"]

        while True:

            yield self.machine_down[resource]

            order = yield self.processes_output[process_deps[0]].get()

            product = order["product"]

            setup_time = 0
            if self.last_process[resource] != process:
                setup_dist = self.resources_config[resource]["setup"].get(
                    "dist", "constant"
                )
                setup_params = self.resources_config[resource]["setup"].get(
                    "params", [0]
                )
                setup_time = self.generate_random_number(setup_dist, setup_params)
                if self.env.now >= self.warmup:
                    self.setups_cout[resource] += 1
                    self.setups_time[resource] += setup_time

            self.last_process[resource] = process

            with self.resources[resource].request() as req:

                yield req

                yield self.env.timeout(setup_time)

                process_time_dist = self.processes_config[process][
                    "processing_time"
                ].get("dist")
                process_time_params = self.processes_config[process][
                    "processing_time"
                ].get("params")

                order_quantity = order.get("quantity")
                # yield self.resources_queue[resource].get(order_quantity)

                start_time = self.env.now

                for part in range(order_quantity):

                    # yield self.wait_action
                    processing_time = self.generate_random_number(
                        process_time_dist, process_time_params
                    )

                    yield self.env.timeout(processing_time)

                    if constraint:
                        self.constraint_buffer_level.remove(product)

                order["processes"][process] = self.env.now

                if final_process:
                    order["finished"] = self.env.now
                    yield self.finished_orders.put(order)
                else:
                    next_resource = self.processes_config[next_process[0]]["resource"]
                    # yield self.resources_queue[next_resource].put(order_quantity)
                    yield self.processes_output[process].put(order)

                end_time = self.env.now
                if self.env.now > self.warmup:
                    self.utilization[resource] += round(end_time - start_time, 8)

    def _measure_product_wip(self, product):
        # TODO define measure metrics
        pass

    def _monitor(self, interval):

        yield self.env.timeout(self.warmup)

        while True:

            total_wip_count = 0
            total_finished_goods_count = 0
            total_inventory_count = 0

            # product monitor
            for product in self.products_config:
                # wip
                wip_product = self.measure_product_wip(product)
                self.wip_product_monitor[product].append(wip_product)
                total_wip_count += wip_product

                # finished goods
                finished_products = self.finished_goods[product].level
                self.finished_goods_monitor[product].append(finished_products)
                total_finished_goods_count += finished_products

                # inventory
                inventory_product = wip_product + finished_products
                self.inventory_product_monitor[product].append(inventory_product)
                total_inventory_count += inventory_product

                # shipping buffer
                shipping_buffer = self.shipping_buffer[product]
                self.shipping_buffer_monitor[product].append(shipping_buffer)
                shipping_buffer_level = self.shipping_buffer_level[product].level
                self.shipping_buffer_level_monitor[product].append(
                    shipping_buffer_level
                )

            # resource monitor
            # for resource in self.resources_queue:
            #     queue_len = self.resources_queue[resource].level
            #     self.resources_queue_monitor[resource].append(queue_len)

            # constraint buffer

            self.constraint_buffer_monitor.append(self.constraint_buffer)
            self.constraint_buffer_level_monitor.append(self.drb_measure_ccr_load())
            self.wip_system_monitor.append(total_wip_count)
            self.finished_goods_system_monitor.append(total_finished_goods_count)
            self.inventory_system_monitor.append(total_inventory_count)

            yield self.env.timeout(interval)

    def _simulation_monitor(self, interval):

        inicio = timeit.default_timer()
        pd.options.display.max_columns = None
        pd.options.display.expand_frame_repr = False
        pd.options.display.float_format = "{:.2f}".format

        while True:

            now = self.env.now

            fim = timeit.default_timer()

            print(f"- - - - - - - - - {round(now,0)} - - - - - - - - -")
            print(f"- - - - - - - - - {round(fim - inicio,2)} - - - - - - - - -")

            print("\t")

            perf_measures = self.measure_performance()
            utilization = self.measure_utilization()
            shipping_buffer = self.measure_shipping_buffer()
            constraint_buffer = self.measure_constraint_buffer()

            print(f"- - - - - - - - - PRODUCTION PERFORMANCE - - - - - - - - -")
            print("\t")
            print(perf_measures.T)
            print("\t")

            print(f"- - - - - - - - - RESOURCE UTILIZATION - - - - - - - - -")
            print("\t")
            print(utilization)
            print("\t")

            print(f"- - - - - - - - - SHIPPING BUFFERS - - - - - - - - -")
            print("\t")
            print(shipping_buffer.T)
            print("\t")

            print(f"- - - - - - - - - CONSTRAINT BUFFER - - - - - - - - -")
            print("\t")
            print(constraint_buffer.T)
            print("\t")

            inicio = fim

            yield self.env.timeout(interval)

    def _create_resources(self) -> None:

        self.resources = {}
        self.utilization = {}
        self.setups_cout = {}
        self.setups_time = {}
        self.resources_queue = {}
        self.resources_queue_monitor = {}
        self.machine_down = {}
        self.machine_tbf_monitor = {}
        self.machine_ttr_monitor = {}
        self.last_process = {}

        for resource in self.resources_config:
            resource_config = self.resources_config.get(resource)
            quantity = resource_config.get("quantity", 1)

            self.resources[resource] = simpy.Resource(self.env, quantity)
            self.resources_queue[resource] = simpy.Container(self.env)

            self.machine_down[resource] = self.env.event()
            self.machine_down[resource].succeed()
            self.last_process[resource] = 0
            self.utilization[resource] = 0
            self.setups_cout[resource] = 0
            self.setups_time[resource] = 0
            self.resources_queue_monitor[resource] = []
            self.machine_tbf_monitor[resource] = []
            self.machine_ttr_monitor[resource] = []
