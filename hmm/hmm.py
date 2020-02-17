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
