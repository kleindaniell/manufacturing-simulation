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
        production_mode: str = "pull",  # TODO: Push system
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
        self._create_process()

    def _create_orders(self):
        self.finished_goods = {}

        self.demand = {}
        self.demand_orders = {}
        self.delivered_ontime = {}
        self.delivered_late = {}
        self.tardiness_products = {}
        self.tardiness_system = []
        for product in self.products_config:
            self.finished_goods[product] = simpy.Container(self.env)

            self.demand[product] = simpy.FilterStore(self.env)
            self.demand_orders[product] = simpy.FilterStore(self.env)

            self.delivered_ontime[product] = simpy.FilterStore(self.env)
            self.delivered_late[product] = simpy.FilterStore(self.env)
            self.tardiness_products[product] = []

        self.finished_orders = simpy.FilterStore(self.env)
        self.orders_to_release = simpy.Store(self.env)
        self.wip_finished = simpy.FilterStore(self.env)
        self.wip_id = 0

    def _get_order_resource_queue(self, resource, method):
        match method:
            case "fifo":
                order = yield self.resources_input[resource].get()
            case "toc_penetration":
                # TODO: implement filter for toc penetration method
                order = yield self.resources_input[resource].get()

        return order

    def _transportation(self, resource):
        while True:
            order = yield self.resources_output[resource].get()

            if order["process_total"] == order["process_finished"]:
                order["finished"] = self.env.now
                yield self.finished_orders.put(order)
            else:
                process_id = order["process_finished"]
                next_resource = self.processes_value_list[process_id]["resource"]
                yield self.resources_input[next_resource].put(order)

    def make_PO(self, product, quantity):
        # make wip order
        production_order = {}
        production_order["id"] = self.wip_id
        production_order["product"] = product
        production_order["schedule"] = 0
        production_order["released"] = -1
        production_order["duedate"] = 0
        production_order["finished"] = False
        production_order["quantity"] = quantity
        production_order["priority"] = 0
        production_order["process_total"] = len(
            self.products_config[product]["processes"]
        )
        production_order["process_finished"] = 0
        production_order["processes"] = {}

        for process in self.products_config[product]["processes"]:
            production_order["processes"][process] = -1
        raw_material = self.products_config[product].get("raw_material")
        production_order["processes"][raw_material] = -1

        self.wip_id += 1

        return production_order

    def _release_PO(self, order, callback):
        product = order["product"]
        quantity = order["quantity"]
        schedule = order["schedule"]
        raw_material = self.products_config[product]["raw_material"]

        if schedule > self.env.now:
            delay = schedule - self.env.now
            yield self.env.timeout(delay)

        order["released"] = self.env.now
        # order["processes"][raw_material] = self.env.now

        yield self.processes_output[raw_material].put(order)

        if callback:
            callback()

        # # Add order to shipping buffer
        # yield self.shipping_buffer_level[product].put(quantity)
        # # Add order to wip monitor
        # self.wip_orders[order["id"]] = order

        # # Add produtc to Constraint buffer
        # if order["constraint"]:
        #     for _ in range(quantity):
        #         self.constraint_buffer_level.append(product)
