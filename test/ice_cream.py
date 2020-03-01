# -*- coding: utf-8 -*-

import jax
import jax.numpy as np

from hmm import HiddenMarkovModel
from hmm import functional as F


def decode_test(hmm: HiddenMarkovModel):  
    print(hmm.decode(np.array([2, 0, 2])))


def likelihood_test(hmm: HiddenMarkovModel):
    likelihood = hmm.likelihood(np.array([2, 0, 2]))
    print(f'Likelihood of {[2, 0, 2]} is {np.exp(likelihood):.3f}')


def likelihood_gradients(hmm: HiddenMarkovModel):
    # Maximize likelihood of [0, 2, 0]
    O = np.array([0, 2, 0])

    def nll(markov, O_seq):
        return -F.likelihood(markov, O_seq)

    lh_fn = jax.value_and_grad(nll)
    
    # Compute the probability without optimizing according to
    # negative log likelihood
    value = hmm.likelihood(O)
    print('Probability of [0, 2, 0]:', np.exp(value))

    for i in range(30):
        value, grad = lh_fn(hmm, O)
        hmm = hmm - grad * 1e-1
    
    # See how the probability has increased after 30 steps
    value = hmm.likelihood(O)
    print('Probability of [0, 2, 0]:', np.exp(value))



def batched_likelihood(key: np.ndarray, hmm: HiddenMarkovModel):
    v_likelihood = jax.vmap(F.likelihood, in_axes=(None, 0))
    
    key, sk = jax.random.split(key)
    Os = jax.random.randint(sk, shape=(4, 3), minval=0, maxval=3)
    Os = np.vstack([Os, np.array([2, 0, 2])])
    print(v_likelihood(hmm, Os))


def sample_test(key, hmm: HiddenMarkovModel):
    # Sample a random sequence of observations
    key, sk = jax.random.split(key)
    print(hmm.sample(key, 3))


def observation_sequence_test(hmm: HiddenMarkovModel):
    
    log_prob = hmm.observations_sequence_proba(
        O=np.array([2, 0, 1]),
        known_Q=np.array([0, 0, 1]))
    print(np.exp(log_prob))


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

    B = np.array([[.0, .5, .5],  # Emission probs being at hot
                  [.5, .4, .1]]) # Emission probs being at cold 

    hmm = HiddenMarkovModel(A=A, B=B, pi=pi)

    observation_sequence_test(hmm)
    sample_test(key, hmm)
    
    likelihood_test(hmm)
    likelihood_gradients(hmm)
    batched_likelihood(key, hmm)

    decode_test(hmm)
    # # draw_test(hmm)
