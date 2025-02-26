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
        self.warmup = warmup
        self.run_until = run_until
        self.monitor_interval = monitor
        self.actions_type = actions
        self.env_mode = env_mode

        # Absolute params
        self.scheduler = self.params["environment"]["scheduler"]
        self.schedule_interval = self.params["environment"]["schedule_interval"]
        self.buffer_adjust_interval = self.params["environment"]["buffer_interval"]
        self.sch_max_action = self.params["environment"]["sch_max_action"]
        self.action_power = self.params["environment"]["action_power"]
        self.wip_multiplier = self.params["environment"]["wip_multiplier"]
        self.fgoods_multiplier = self.params["environment"]["fgoods_multiplier"]
        self.reward_add = self.params["environment"]["reward_add"]
        # create environment
        