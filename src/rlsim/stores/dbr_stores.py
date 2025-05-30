from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

import simpy

from rlsim.engine.control import Stores


class DBR_stores(Stores):
    def __init__(
        self,
        env: simpy.Environment,
        resources: dict,
        products: dict,
        cb_start: int,
    ):
        super().__init__(env, resources, products)

        self._create_constraint_buffer(cb_start)
        self._create_shipping_buffers()

    def _create_constraint_buffer(self, cb_start):
        # Constraint buffers
        self.constraint_buffer = cb_start
        self.constraint_buffer_level = 0

    def _create_shipping_buffers(self):
        # Shipping_buffer
        self.shipping_buffer = {}
        self.shipping_buffer_level = {}

        for product in self.products:
            self.shipping_buffer[product] = self.products[product].get(
                "shipping_buffer", 0
            )
            self.shipping_buffer_level[product] = simpy.Container(self.env)

    def _define_constraint(self):
        
        