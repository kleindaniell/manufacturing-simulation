import random
import numpy as np
from scipy.stats import gamma, erlang


def random_number(distribution, params) -> float:
    value = 0
    if distribution == "constant":
        value = params[0]
    elif distribution == "uniform":
        c = params[1] * 2 * np.sqrt(3)
        a = params[0] - (c / 2)
        b = params[0] + (c / 2)
        value = random.uniform(a, b)
    elif distribution == "gamma":
        k = params[0] ** 2 / params[1] ** 2
        theta = params[1] ** 2 / params[0]
        value = random.gammavariate(k, theta)
    elif distribution == "erlang":
        k = params[0] ** 2 / params[1] ** 2
        theta = params[1] ** 2 / params[0]
        value = random.gammavariate(k, theta)
    elif distribution == "expo":
        value = random.expovariate(1 / params[0])
    elif distribution == "normal":
        value = random.normalvariate(params[0], params[1])
    else:
        value = 0

    return value
