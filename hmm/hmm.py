# -*- coding: utf-8 -*-

import itertools
from typing import Mapping, NamedTuple

import jax
import jax.numpy as np

from hmm import functional as F


class HiddenMarkovModel(NamedTuple):

    """
    Class to represent a Hidden Markov Model (HMM)

    Parameters
    ----------
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
    A: np.ndarray = None
    B: np.ndarray = None 
    pi: np.ndarray = None

    @property
    def normalized_A(self) -> np.ndarray:
        return jax.nn.log_softmax(self.A, axis=-1)
    
    @property
    def normalized_B(self) -> np.ndarray:
        return jax.nn.log_softmax(self.B, axis=-1)
    
    @property
    def normalized_pi(self) -> np.ndarray:
        return jax.nn.log_softmax(self.pi, axis=-1)
     
    @classmethod
    def random_init(cls, 
                    key: np.ndarray, 
                    n_hidden_states: int, 
                    n_observations: int) -> 'HiddenMarkovModel':
        """
        Class method to create a Hidden Markov Model with randomly initialized
        parameters. The parameters that are randomly initialized are the 
        transition matrix A, the emission probabilities B and the state prior 
        distribution pi
        
        Parameters
        ----------
        key: np.ndarray
            Random seed used to initialize the parameters
        n_hidden_states: int
            Number of possible hidden states
        n_observations: np.ndarray
            Number of possible observations
        """
        key, A_key, B_key, pi_key = jax.random.split(key, 4)

        init_fn = jax.nn.initializers.uniform()
        A = init_fn(A_key, shape=(n_hidden_states,) * 2)
        B = init_fn(B_key, shape=(n_hidden_states, n_observations))
        pi = init_fn(pi_key, shape=(n_hidden_states,))

        return cls(A=A, B=B, pi=pi)
    
            
    def observations_sequence_proba(self, 
                                    O: np.ndarray,
                                    known_Q: np.ndarray) -> float:
        """
        Given that A, B and the sequence of hidden states are known, 
        we can compute the probability of observing the given observation 
        sequence (O)

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
        return F.observations_sequence_proba(self, O, known_Q)
    
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
        return F.sample(self, key, timesteps)

    def likelihood(self, O: np.ndarray) -> float:
        """
        Implementation of forward algorithm to compute the likelihood of 
        the sequence of observations O

        Parameters
        ----------
        O: np.ndarray
            Sequence of observations
        
        reduce: bool, default True
            If set to true the method just returns the likelihood of the 
            sequence, otherwise, it returns the whole trellis used to compute the
            sequence probability
        Returns
        -------
        float
            Likelihood of observing the specified sequence of observations
        """
        return F.likelihood(self, O)

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
        return F.decode(self, O)

    def __add__(self, other: 'HiddenMarkovModel')-> 'HiddenMarkovModel':
        return HiddenMarkovModel(
            A=self.A + other.A,
            B=self.B + other.B,
            pi=self.pi + other.pi)
    
    def __sub__(self, other: 'HiddenMarkovModel') -> 'HiddenMarkovModel':
        return HiddenMarkovModel(
            A=self.A - other.A,
            B=self.B - other.B,
            pi=self.pi - other.pi)

    def __mul__(self, other: float) -> 'HiddenMarkovModel':
        assert isinstance(other, float), 'Only scalar multiplication is allowed'

        return HiddenMarkovModel(
            A=self.A * other,
            B=self.B * other,
            pi=self.pi * other)

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

        Q = np.arange(self.A.shape[0])
        O = np.arange(self.B.shape[1])

        # Add hidden states nodes
        with graph.subgraph(name='hidden_states') as c:
            c.node_attr.update(style='filled', color='lightblue')
            for q in Q:
                c.node('q-' + str(q), 
                       label=(Q_idx2name and str(Q_idx2name[q])))
        
        # Draw the observations
        with graph.subgraph(name='observations') as obs_g:
            obs_g.graph_attr.update(rankdir='LR')
            obs_g.node_attr.update(style='filled', color='#d3f0ce')
            for o in O:
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
