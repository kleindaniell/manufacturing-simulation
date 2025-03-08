import simpy
from control import Info, Controller, ProductionOrder


class Scheduler:
    def __init__(self, env: simpy.Environment, info: Info):
        self.env = env
        self.info = info
        self.controller = Controller(self.env, self.info)

    def release_uniform(self):
        self.controller.release_production_order()
    
        