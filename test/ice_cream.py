# -*- coding: utf-8 -*-

import jax.numpy as np

from hmm import HiddenMarkovModel


def ice_cream_test():
    Q_names = ['hot', 'cold']
    
    Q = [0, 1]
    pi = [.8, .2]
    O = list(range(3)) # [1, 2, 3]
    
    A = [[.6, .4],
         [.5, .5]]

    B = [[.2, .4, .4], # Emission probs being at hot
         [.5, .4, .1]] # Emission probs being at cold 

    hmm = HiddenMarkovModel(
        Q=np.array(Q),
        O=np.array(O),
        A=np.array(A),
        B=np.array(B),
        pi=np.array(pi))
    
    prob = hmm.observations_sequence_proba(
        O=np.array([2, 0, 1]),
        known_Q=np.array([0, 0, 1]))
    print(prob)

    prob = hmm.observations_sequence_proba(
        O=np.array([2]),
        known_Q=np.array([0]))
    print(prob)


if __name__ == "__main__":
    ice_cream_test()