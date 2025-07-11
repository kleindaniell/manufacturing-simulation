import random
from typing import List, Union, Any, Literal
from pathlib import Path
import numpy as np
import yaml


class DistributionGenerator:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def random_number(
        self,
        distribution: Literal[
            "constant", "uniform", "gamma", "erlang", "expo", "normal"
        ],
        params: List[Any],
    ) -> float:
        if distribution == "constant":
            value = params[0]
        elif distribution == "uniform":
            c = params[1] * 2 * np.sqrt(3)
            a = params[0] - (c / 2)
            b = params[0] + (c / 2)
            value = self.rng.uniform(a, b)
        elif distribution == "gamma":
            k = params[0] ** 2 / params[1] ** 2
            theta = params[1] ** 2 / params[0]
            value = self.rng.gammavariate(k, theta)
        elif distribution == "erlang":
            k = params[0] ** 2 / params[1] ** 2
            theta = params[1] ** 2 / params[0]
            value = self.rng.gammavariate(k, theta)
        elif distribution == "expo":
            value = self.rng.expovariate(1 / params[0])
        elif distribution == "normal":
            value = self.rng.normalvariate(params[0], params[1])
        else:
            raise ValueError(f"Unknowh distribution type {distribution}")

        return max(0, np.float32(value))

    def random_int(self, start=0, end=999999) -> int:
        return self.rng.randint(a=start, b=end)


def load_yaml(yaml_file_path: Union[str, Path]):
    """Load configuration from YAML files"""
    if not isinstance(yaml_file_path, Path):
        yaml_file_path = Path(yaml_file_path)

    if yaml_file_path.is_file():
        with open(yaml_file_path, "r") as file:
            file_data = yaml.safe_load(file)
            return file_data
    else:
        raise ValueError("File path not provided")
