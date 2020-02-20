# -*- coding: utf-8 -*-

import itertools
from typing import Mapping

import jax
import jax.numpy as np

from hmm import utils


class HiddenMarkovModel(object):

    """
    Class to represent a Hidden Markov Model (HMM)

    Parameters
    ----------
    Q: np.ndarray
        Set of hidden states
    O: np.ndarray
        Set of possible observations
    A: np.ndarray, default None
        Transmission probabilities. The transmission probabilities are 
        represented using a matrix of shape [len(Q), len(Q)] and the element
        aij contains the probability of moving from the state Qi to the hidden
        state Qj
    B: np.ndarray, default None
        Emission probabilities, emission probabilities are defined as a tensor 
        of shape [len(Q), len(O)] where each row `i` contains the probability 
        distribution of observations while being at Qi hidden state
    pi: np.ndarray
        Probability distribution of starting at a concrete hidden state Q
    """
    def __init__(self, 
                 Q: np.ndarray,
                 O: np.ndarray,
                 A: np.ndarray = None,
                 B: np.ndarray = None,
                 pi: np.ndarray = None):
        self.Q = Q
        self.O = O
        self.A = A
        self.B = B
        self.Q = Q
        self.pi = pi
    
    def random_init(self, key: np.ndarray):
        """
        Randomly starts all the parameters that where not set during the 
        instance creation
        
        Parameters
        ----------
        key: None
            Random seed used to initialize the parameters
        
        Examples
        --------
        >>> hmm = HiddenMarkovModel(Q=...)
        >>> key = jax.random.PRNGKey(0)
        >>> hmm.init_random(key) # Here A, B are randomly initialized
        """
        if self.A is None:
            key, subkey = jax.random.split(key)
            self.A = jax.random.uniform(
                key, shape=(self.Q.shape[0],) * 2)
        
        if self.B is None:
            key, subkey = jax.random.split(key)
            self.B = jax.random.uniform(
                key, shape=(self.Q.shape[0], self.O.shape[0]) * 2)
        
        return self
            
    def observations_sequence_proba(self, 
                                    O: np.ndarray,
                                    known_Q: np.ndarray) -> float:
        """
        Given that A, B and the sequence hidden states are known, we can compute
        the probability of observing the given observation sequence (O)

        Parameters
        ----------
        O: np.ndarray
            Sequence of observations over time
        known_Q: np.ndarray
            Sequence of known hidden states. len(O) must be equal to (known_Q)
        
        Retruns
        -------
        float
            The probability of observing O
        """
        start_prob = self.pi[known_Q[0]]
        if known_Q.shape[0] > 1:
            transitions = np.array(list(zip(known_Q[:-1], known_Q[1:])))
            known_Q_prob = start_prob * np.prod(self.A[transitions])
        else:
            known_Q_prob = start_prob
        return known_Q_prob * np.prod(self.B[known_Q, O])
    
    def sample(self, key: np.ndarray, timesteps: int) -> np.ndarray:
        """
        Sample an observation at each timestep according to the defined
        probabilites

        Parameters
        ----------
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
        def sample_observation(key, qt):
            emission_probs = self.B[qt]
            return utils.sample(key, self.O, emission_probs)
        
        def sample_qt(key, qt_1):
            transition_probs = self.A[qt_1]
            return utils.sample(key, self.Q, transition_probs)
            
        O = []

        # Sample the initial hidden state according to pi prob distribution
        key, subkey = jax.random.split(key)
        q_0 = utils.sample(subkey, self.Q, self.pi)
        qt = q_0
        for t in range(timesteps):
            key, b_key, q_key = jax.random.split(key, 3)
            # At each hidden state qt, sample one observation according to 
            # emission prob B
            O.append(sample_observation(b_key, qt))
            # Sample the next hidden state
            qt = sample_qt(q_key, qt)
        
        return np.array(O)

    def likelihood(self, O: np.ndarray) -> float:
        """
        Implementation of forward algorithm to compute the likelihood of 
        the sequence of observations O

        Parameters
        ----------
        O: np.ndarray
            Sequence of observations
        
        Returns
        -------
        float
            Likelihood of observing the specified sequence of observations
        """
        timesteps = len(O)
        alpha = np.zeros((len(self.Q), timesteps))

        # Compute the probabilities at timestep = 0
        prob = self.pi[self.Q] * self.B[self.Q, O[0]]
        alpha = jax.ops.index_update(alpha, jax.ops.index[self.Q, 0], prob)

        for t in range(1, timesteps):
            for qt in self.Q:
                new_alpha = (alpha[self.Q, t - 1] * 
                             self.A[self.Q, qt] * self.B[qt, O[t]])
                new_alpha = np.sum(new_alpha)
                alpha = jax.ops.index_add(
                    alpha, jax.ops.index[qt, t], new_alpha)

        return np.sum(alpha[:, -1])
    
    def decode(self, O: np.ndarray) -> np.ndarray:
        """
        Find the sequence of hidden states that maximizes the 
        likelihood of observing the observervation sequence O

        This method implements the Viterbi algorithm.

        Parameters
        ----------
        O: np.ndarray
            Sequence of observations 
        
        Returns
        -------
        Tuple[float, np.ndarray]
            Returns a tuple containing the most probable sequence probability 
            and the sequence of hidden states
        """
        timesteps = len(O)
        v = np.zeros((len(self.Q), timesteps))
        backpointer = np.zeros((len(self.Q), timesteps))

        # Compute the probabilities at timestep = 0
        prob = self.pi[self.Q] * self.B[self.Q, O[0]]
        v = jax.ops.index_update(v, jax.ops.index[self.Q, 0], prob)

        for t in range(1, timesteps):
            for qt in self.Q:
                new_vt = (v[self.Q, t - 1] * 
                          self.A[self.Q, qt] * self.B[qt, O[t]])
                v = jax.ops.index_update(
                    v, jax.ops.index[qt, t], np.max(new_vt))

                best_path = np.argmax(new_vt)
                backpointer = jax.ops.index_update(
                    backpointer, jax.ops.index[qt, t], best_path)
        
        best_prob = np.max(v[:, -1])
        best_path_pointer = np.argmax(v[:, -1])

        return best_prob, np.array(
            backpointer[best_path_pointer, 1:].tolist() + [best_path_pointer])

    def draw(self,
             Q_idx2name: Mapping[int, str] = None,
             O_idx2name: Mapping[int, str] = None) -> 'Digraph':
        """
        Draw the Hidden Markov Model using Graphviz

        Parameters
        ----------
        Q_idx2name: Mapping[int, str], default None
            Mapping from getting the name based on the hidden state idx
        O_idx2name: Mapping[int, str], default None
            Mapping to get the name based on the observation idx
        
        Returns
        -------
        graphviz.Digraph
            The resulting graph of drawing the hidden markov model
            The hidden states are filled with blue, and the observations
            are filled with green.
        """
        # We don't want this dipendency as a must
        from graphviz import Digraph
        
        graph = Digraph(name='Hidden Markov Model', format='svg')

        # Add hidden states nodes
        with graph.subgraph(name='hidden_states') as c:
            c.node_attr.update(style='filled', color='lightblue')
            for q in self.Q:
                c.node('q-' + str(q), 
                       label=(Q_idx2name and str(Q_idx2name[q])))
        
        # Draw the observations
        with graph.subgraph(name='observations') as obs_g:
            obs_g.graph_attr.update(rankdir='LR')
            obs_g.node_attr.update(style='filled', color='#d3f0ce')
            for o in self.O:
                 obs_g.node('o-' + str(o), 
                            label=(O_idx2name and str(O_idx2name[o])))

        # Draw the starting node
        graph.node('pi')
        for i, start_prob in enumerate(self.pi):
            graph.edge('pi', 'q-' + str(i), label=f'{start_prob:.2f}')

        matrix_it = range(self.B.shape[0]), range(self.B.shape[1])
        for i, j in itertools.product(*matrix_it): 
            e_p = self.B[i, j]
            if e_p > 0:
                graph.edge('q-' + str(i), 
                           'o-' + str(j), label=f'{e_p:.2f}')

        matrix_it = range(self.A.shape[0]), range(self.A.shape[1])
        for i, j in itertools.product(*matrix_it):
            p = self.A[i, j]
            if p > 0:
                graph.edge('q-' + str(i), 'q-' + str(j), label=f'{p:.2f}')
            
        return graph
