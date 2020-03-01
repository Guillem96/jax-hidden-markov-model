# -*- coding: utf-8 -*-

import jax
import jax.numpy as np

from hmm import utils


def _logdotexp(A, B):
    max_A = np.max(A)
    max_B = np.max(B)
    C = np.dot(np.exp(A - max_A), np.exp(B - max_B))
    C = np.log(C)
    C += max_A + max_B
    return C

def observations_sequence_proba(hmm: 'HiddenMarkovModel', 
                                O: np.ndarray,
                                known_Q: np.ndarray) -> float:
    """
    Given that A, B and the sequence of hidden states are known, 
    we can compute the probability of observing the given observation 
    sequence (O)

    Parameters
    ----------
    hmm: HiddenMarkovModel
        NamedTuple representing an HMM
    O: np.ndarray
        Sequence of observations over time
    known_Q: np.ndarray
        Sequence of known hidden states. len(O) must be equal to (known_Q)
    
    Retruns
    -------
    float
        The probability of observing O
    """
    start_prob = hmm.normalized_pi[known_Q[0]]
    if known_Q.shape[0] > 1:
        transitions = np.array(list(zip(known_Q[:-1], known_Q[1:])))
        known_Q_prob = start_prob + np.sum(hmm.normalized_A[transitions])
    else:
        known_Q_prob = start_prob
    return known_Q_prob + np.sum(hmm.normalized_B[known_Q, O])


def sample(hmm: 'HiddenMarkovModel', 
           key: np.ndarray, timesteps: int) -> np.ndarray:
    """
    Sample an observation at each timestep according to the defined
    probabilites

    Parameters
    ----------
    hmm: HiddenMarkovModel
        NamedTuple representing an HMM
    key: np.ndarray
        Random seed to do the sampling
    timesteps: int
        Number of observations to sample
    
    Returns
    -------
    np.ndarray
        An array of shape [timesteps,] containing the observations at each
        timesteps
    """
   
    hmm_Q = np.arange(hmm.A.shape[0])
    hmm_O = np.arange(hmm.B.shape[1])
    
    A = hmm.normalized_A
    B = hmm.normalized_B
    O = []

    # Sample the initial hidden state according to pi prob distribution
    key, subkey = jax.random.split(key)
    q_0 = utils.sample(subkey, hmm_Q, hmm.normalized_pi)

    qt = q_0
    for t in range(timesteps):
        key, b_key, q_key = jax.random.split(key, 3)
        # At each hidden state qt, sample one observation according to 
        # emission prob B
        O.append(utils.sample(b_key, hmm_O, B[qt]))
        # Sample the next hidden state
        qt = utils.sample(q_key, hmm_O, A[qt])
    
    return np.array(O)


def likelihood(hmm: 'HiddenMarkovModel', O: np.ndarray) -> float:
    """
    Implementation of forward algorithm to compute the likelihood of 
    the sequence of observations O

    Parameters
    ----------
    hmm: HiddenMarkovModel
        NamedTuple representing an HMM
    O: np.ndarray
        Sequence of observations

    Returns
    -------
    float
        Likelihood of observing the specified sequence of observations
    """
    hmm_Q = np.arange(hmm.A.shape[0])
    A = hmm.normalized_A
    B = hmm.normalized_B

    timesteps = len(O)
    alpha = np.zeros((len(hmm_Q), timesteps))
  
    # Compute the probabilities at timestep = 0
    prob = hmm.normalized_pi[hmm_Q] + B[hmm_Q, O[0]]
    alpha = jax.ops.index_update(alpha, jax.ops.index[hmm_Q, 0], prob)

    for t in range(1, timesteps):
        new_alpha = _logdotexp(alpha[:, t - 1], A) + B[:, O[t]]
        alpha = jax.ops.index_update(alpha, jax.ops.index[:, t], new_alpha)

    return jax.scipy.special.logsumexp(alpha[:, -1])


def decode(hmm: 'HiddenMarkovModel', O: np.ndarray) -> np.ndarray:
    """
    Find the sequence of hidden states that maximizes the 
    likelihood of observing the observervation sequence O

    This method implements the Viterbi algorithm.

    Parameters
    ----------
    hmm: HiddenMarkovModel
        NamedTuple representing an HMM
    O: np.ndarray
        Sequence of observations 
    
    Returns
    -------
    Tuple[float, np.ndarray]
        Returns a tuple containing the most probable sequence probability 
        and the sequence of hidden states
    """
    hmm_Q = np.arange(hmm.A.shape[0])
    hmm_O = np.arange(hmm.B.shape[1])
    A = hmm.normalized_A
    B = hmm.normalized_B

    timesteps = len(O)
    v = np.zeros((len(hmm_Q), timesteps))
    backpointer = np.zeros((len(hmm_Q), timesteps), dtype='int32')

    # Compute the probabilities at timestep = 0
    prob = hmm.normalized_pi + B[:, O[0]]
    v = jax.ops.index_update(v, jax.ops.index[:, 0], prob)

    for t in range(1, timesteps):
        for qt in hmm_Q:
            new_vt = v[:, t - 1] + A[:, qt] + B[qt, O[t]]
            v = jax.ops.index_update(
                v, jax.ops.index[qt, t], np.max(new_vt))

            best_path = np.argmax(new_vt)
            backpointer = jax.ops.index_update(
                backpointer, jax.ops.index[qt, t], best_path)
    
    best_prob = np.max(v[:, -1])
    best_path_pointer = np.argmax(v[:, -1])

    path = [best_path_pointer]
    pointer = best_path_pointer

    # Backtrack the most probable state
    for t in range(timesteps - 1, 0, - 1):
        state = backpointer[pointer, t]
        path.insert(0, state)
        pointer = state

    return best_prob, np.array(path)
