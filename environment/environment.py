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
    def  __init__(self,
                  run_until: int,
                  resources_cfg: dict,
                  products_cfg: dict,
                  raw_material_cfg: dict,
                  schedule_interva: int,
                  buffer_adjust_interval: int,
                  set_constraint: int=None,
                  monitor: bool=False,
                  warmup: bool=False,
                  actions: str="discret",
                  env_mode: str="simulation",
                  seed: int=None,
                  ):
        super().__init__()
        random.seed(seed)
        self.env = simpy.Environment()
        
        # Parameters
        self.resources = resources_cfg
        self.products = products_cfg
        self.raw_material = raw_material_cfg
        self.warmup = warmup
        self.run_until = run_until
        self.monitor_interval = monitor
        self.actions_type = actions
        self.env_mode = env_mode
        self.schedule_interva = schedule_interva
        self.buffer_adjust_interval = buffer_adjust_interval
        self.constraint = set_constraint
        
        
        