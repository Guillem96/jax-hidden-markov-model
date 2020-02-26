# -*- coding: utf-8 -*-

import jax
import jax.numpy as np

from hmm import HiddenMarkovModel


def decode_test(hmm: HiddenMarkovModel):  
    print(hmm.decode(np.array([2, 0, 2])))
    # print(f'Decodification of {[2, 0, 2]} is {likelihood:.3f}')

def likelihood_test(hmm: HiddenMarkovModel):
    likelihood = hmm.likelihood(np.array([2, 0, 2]))
    print(f'Likelihood of {[2, 0, 2]} is {likelihood:.3f}')


def sample_test(key, hmm: HiddenMarkovModel):
    # Sample a random sequence of observations
    key, sk = jax.random.split(key)
    print(hmm.sample(key, 3))


def observation_sequence_test(hmm: HiddenMarkovModel):
    
    prob = hmm.observations_sequence_proba(
        O=np.array([2, 0, 1]),
        known_Q=np.array([0, 0, 1]))
    print(prob)


def draw_test(hmm: HiddenMarkovModel):

    hmm.draw(
        Q_idx2name=['hot', 'cold'],
        O_idx2name=[1, 2, 3]).render(format='png', view=True)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    Q_names = ['hot', 'cold']
    
    pi = np.array([.8, .2])
    
    A = np.array([[.6, .4],
                  [.5, .5]])

    B = np.array([[.2, .4, .4],  # Emission probs being at hot
                  [.5, .4, .1]]) # Emission probs being at cold 

    hmm = HiddenMarkovModel(A=A, B=B, pi=pi)

    observation_sequence_test(hmm)
    sample_test(key, hmm)
    likelihood_test(hmm)
    decode_test(hmm)
    # draw_test(hmm)
