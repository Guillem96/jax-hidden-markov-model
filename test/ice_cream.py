# -*- coding: utf-8 -*-

import jax
import jax.numpy as np

from hmm import HiddenMarkovModel


def ice_cream_test():
    key = jax.random.PRNGKey(0)

    Q_names = ['hot', 'cold']
    Q = np.array([0, 1])

    pi = np.array([.8, .2])
    O = np.arange(3) # [1, 2, 3]
    
    A = np.array([[.6, .4],
                  [.5, .5]])

    B = np.array([[.2, .4, .4],  # Emission probs being at hot
                  [.5, .4, .1]]) # Emission probs being at cold 

    hmm = HiddenMarkovModel(Q=Q, O=O, A=A, B=B, pi=pi)

    prob = hmm.observations_sequence_proba(
        O=np.array([2, 0, 1]),
        known_Q=np.array([0, 0, 1]))
    print(prob)

    # Sample a random sequence of observations
    key, sk = jax.random.split(key)
    print(hmm.sample(key, 3))


if __name__ == "__main__":
    ice_cream_test()