import random
import numpy as np


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
