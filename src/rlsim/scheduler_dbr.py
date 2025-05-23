import simpy

from rlsim.control import Stores, ProductionOrder
from rlsim.monitor import Monitor
from rlsim.scheduler import Scheduler


class DBR(Scheduler):
    def __init__(self, store, interval):
        super().__init__(store, interval)

    def _scheduler(self, interval):
        return super()._scheduler(interval)
