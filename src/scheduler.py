import simpy
from control import Stores, Controller, ProductionOrder


class Scheduler:
    def __init__(self, store: Stores, interval: int):
        self.store: Stores = store
        self.env: simpy.Environment = store.env
        self.interval = interval
        self.controller = Controller(self.env, self.store)
        
        self.env.process(self.release_uniform())


    def release_uniform(self):
        order = ProductionOrder(self.env, self.store, "produto01")
        print(order.to_dict())
        order.release()
        yield self.env.timeout(self.interval)

    
        