import random
from typing import List, Union, Any, Literal, Tuple
from pathlib import Path
import numpy as np
import yaml


def _uniform_bounds(params: List[Any]) -> Tuple[float, float]:
    c = params[1] * 2 * np.sqrt(3)
    a = params[0] - (c / 2)
    b = params[0] + (c / 2)
    return a, b


def _gamma_shape_scale(params: List[Any]) -> Tuple[float, float]:
    k = params[0] ** 2 / params[1] ** 2
    theta = params[1] ** 2 / params[0]
    return k, theta


class DistributionGenerator:
    """Random draws for simulation distributions.

    Scalar draws use ``random.Random``. Batched sums (e.g. processing time for
    many units) use ``numpy.random.Generator`` so the stdlib RNG is not
    advanced by large per-order loops. Same integer ``seed`` is deterministic
    for both streams; trajectories differ from an older implementation that drew
    all processing times from the stdlib RNG only.
    """

    def __init__(self, seed):
        self.rng = random.Random(seed)
        # Deterministic NumPy stream, independent of ``self.rng`` (see class docstring).
        # ``seed`` can be None during env construction (e.g. SubprocVecEnv on spawn).
        # In that case use OS entropy, matching random.Random(None) semantics.
        if seed is None:
            seed_seq = np.random.SeedSequence()
        else:
            seed_seq = np.random.SeedSequence([int(seed), 0x4E505F42])
        self.np_rng = np.random.default_rng(seed_seq)

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
            a, b = _uniform_bounds(params)
            value = self.rng.uniform(a, b)
        elif distribution == "gamma":
            k, theta = _gamma_shape_scale(params)
            value = self.rng.gammavariate(k, theta)
        elif distribution == "erlang":
            k, theta = _gamma_shape_scale(params)
            value = self.rng.gammavariate(k, theta)
        elif distribution == "expo":
            value = self.rng.expovariate(1 / params[0])
        elif distribution == "normal":
            value = self.rng.normalvariate(params[0], params[1])
        else:
            raise ValueError(f"Unknowh distribution type {distribution}")

        return max(0, np.float32(value))

    def sum_random_numbers_batch(
        self,
        distribution: Literal[
            "constant", "uniform", "gamma", "erlang", "expo", "normal"
        ],
        params: List[Any],
        count: int,
    ) -> float:
        """Sum of ``count`` i.i.d. samples; uses ``np_rng``, not ``random.Random``."""
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return 0.0

        if distribution == "constant":
            total = count * params[0]
        elif distribution == "uniform":
            a, b = _uniform_bounds(params)
            total = self.np_rng.uniform(a, b, size=count).sum()
        elif distribution == "gamma":
            k, theta = _gamma_shape_scale(params)
            total = self.np_rng.gamma(k, theta, size=count).sum()
        elif distribution == "erlang":
            k, theta = _gamma_shape_scale(params)
            total = self.np_rng.gamma(k, theta, size=count).sum()
        elif distribution == "expo":
            total = self.np_rng.exponential(scale=params[0], size=count).sum()
        elif distribution == "normal":
            samples = self.np_rng.normal(params[0], params[1], size=count)
            total = np.maximum(0, np.float32(samples)).sum()
        else:
            raise ValueError(f"Unknowh distribution type {distribution}")

        return float(total)

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
