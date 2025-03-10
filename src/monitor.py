import simpy
from control import Stores

import pandas as pd

class Monitor():
    def __init__(
            self,
            stores: Stores,
            interval: int = 72,
            warmup: int = 0,
            print_type: str = "all"
    ):
        self.stores = stores
        self.env: simpy.Environment = stores.env
        self.interval = interval
        self.warmup = warmup
        self.print_type = print_type

        self.env.process(self.run())
    
    def run(self):

        yield self.env.timeout(self.warmup)
        
        while True:
            orders = self.measure_total_wip()
            print(len(orders))
            self.env.timeout(self.interval)

    
    def measure_total_wip(self):
        orders = []
        for resource in self.stores.resources:
            
            orders.extend(self.stores.resource_output[resource].items)
            orders.extend(self.stores.resource_input[resource].items)
        return orders
            



