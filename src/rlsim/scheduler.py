import simpy
from control import Stores, ProductionOrder


class Scheduler:
    def __init__(self, store: Stores, interval: int):
        self.stores = store
        self.env: simpy.Environment = store.env
        self.interval = interval

        self.env.process(self.release_uniform())

    def release_uniform(self):
        while True:
            order1 = ProductionOrder(self.stores, "produto01")
            order2 = ProductionOrder(self.stores, "produto02")
            # print(order.to_dict())
            order1.release()
            # order1.release()
            order2.release()
            yield self.env.timeout(self.interval)
