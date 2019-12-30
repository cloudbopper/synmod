"""Generator base class"""

from abc import ABC
from functools import partial

import numpy as np
from scipy.stats import bernoulli

from synmod.constants import CONTINUOUS, BINARY, CATEGORICAL, ORDINAL


class Generator(ABC):
    """Generator base class"""
    def __init__(self, rng, feature_type):
        self._rng = rng
        self._feature_type = feature_type

    def sample(self, sequence_length):
        """Sample sequence of given length from generator"""


class BernoulliProcess(Generator):
    """Bernoulli process generator"""
    def __init__(self, rng, feature_type, **kwargs):
        super().__init__(rng, feature_type)
        self._p = kwargs.get("p", self._rng.uniform(0.01, 0.99))

    def sample(self, sequence_length):
        return bernoulli.rvs(p=self._p, size=sequence_length, random_state=self._rng)


# pylint: disable = invalid-name, pointless-statement
class MarkovChain(Generator):
    """Markov chain generator"""

    class State():
        """Markov chain state"""
        # pylint: disable = protected-access
        def __init__(self, chain, index):
            self._chain = chain  # Parent Markov chain
            self._index = index
            self._p = None  # Transition probabilities from state
            self.sample = None  # Function to sample from state distribution
            self._gen_distributions()

        def _gen_distributions(self):
            """Generate state transition and sampling distributions"""
            feature_type = self._chain._feature_type
            rng = self._chain._rng
            self._p = rng.uniform(size=self._chain._n_states)
            if feature_type == CONTINUOUS:
                self.sample = partial(rng.normal, rng.uniform(), rng.uniform() * 0.05)
            elif feature_type in {BINARY, CATEGORICAL, ORDINAL}:
                self.sample = lambda: self._index
                if feature_type == ORDINAL:
                    # dissallow transitions to non-consecutive states:
                    # TODO: more elegant solution?
                    mask = np.zeros(self._chain._n_states)
                    mask[max(0, self._index - 1): min(self._chain._n_states, self._index + 2)] = 1
                    self._p *= mask
            else:
                raise ValueError("Feature type invalid: {0}".format(feature_type))
            self._p /= self._p.sum()  # normalize transition probabilities

        def transition(self):
            """Transition to next state"""
            return self._chain._rng.choice(self._chain._states, p=self._p)


    def __init__(self, rng, feature_type, **kwargs):
        super().__init__(rng, feature_type)
        self._n_states = kwargs.get("n_states", self._rng.integers(2, 5, endpoint=True))
        if self._feature_type == BINARY:
            self._n_states = 2
        self._states = [self.State(self, index) for index in range(self._n_states)]


    def sample(self, sequence_length):
        cur_state = self._rng.choice(self._states)  # initial state
        sequence = np.empty(sequence_length)
        for timestep in range(sequence_length):
            sequence[timestep] = cur_state.sample()
            cur_state = cur_state.transition()
        return sequence
