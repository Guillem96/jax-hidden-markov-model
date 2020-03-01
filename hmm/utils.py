# -*- coding: utf-8 -*-

from typing import Tuple

import jax
import jax.numpy as np


def sample(key: np.ndarray,
           outcomes: np.ndarray,
           prob_distribution: np.ndarray, 
           shape: Tuple[int] = (1,)) -> int:
           
    idx = jax.random.categorical(key, prob_distribution, shape=(1,))
    return outcomes[idx]
