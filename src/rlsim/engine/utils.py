import random
from typing import List, Union, Any
from pathlib import Path
import numpy as np
import yaml


class Distribution:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def random_number(self, distribution: str, params: list) -> float:
        value = 0
        match distribution:
            case "constant":
                value = params[0]
            case "uniform":
                c = params[1] * 2 * np.sqrt(3)
                a = params[0] - (c / 2)
                b = params[0] + (c / 2)
                value = self.rng.uniform(a, b)
            case "gamma":
                k = params[0] ** 2 / params[1] ** 2
                theta = params[1] ** 2 / params[0]
                value = self.rng.gammavariate(k, theta)
            case "erlang":
                k = params[0] ** 2 / params[1] ** 2
                theta = params[1] ** 2 / params[0]
                value = self.rng.gammavariate(k, theta)
            case "expo":
                value = self.rng.expovariate(1 / params[0])
            case "normal":
                value = self.rng.normalvariate(params[0], params[1])
            case TypeError:
                value = 0

        return np.float32(value)


class DistributionGenerator:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def random_number(self, distribution: str, params: List[Any]) -> float:
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

        return np.float32(value)


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
