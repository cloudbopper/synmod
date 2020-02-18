"""Generator base class"""

from abc import ABC

import numpy as np
from scipy.stats import bernoulli

from synmod.constants import CONTINUOUS, BINARY, CATEGORICAL, ORDINAL

IN_WINDOW = "in-window"
OUT_WINDOW = "out-window"


class Generator(ABC):
    """Generator base class"""
    def __init__(self, rng, feature_type, window):
        self._rng = rng
        self._feature_type = feature_type
        self._window = window

    def sample(self, sequence_length):
        """Sample sequence of given length from generator"""


class BernoulliProcess(Generator):
    """Bernoulli process generator"""
    def __init__(self, rng, feature_type, window, **kwargs):
        super().__init__(rng, feature_type, window)
        self._p = kwargs.get("p", self._rng.uniform(0.01, 0.99))

    def sample(self, sequence_length):
        return bernoulli.rvs(p=self._p, size=sequence_length, random_state=self._rng)


# pylint: disable = invalid-name, pointless-statement
class MarkovChain(Generator):
    """Markov chain generator"""

    class State():
        """Markov chain state"""
        # pylint: disable = protected-access
        def __init__(self, chain, index, state_type):
            self._chain = chain  # Parent Markov chain
            self._index = index  # state identifier
            self._state_type = state_type  # state type - in-window vs. out-window
            self._p = None  # Transition probabilities from state
            self._states = None  # States to transition to
            self.sample = None  # Function to sample from state distribution

        def gen_distributions(self):
            """Generate state transition and sampling distributions"""
            feature_type = self._chain._feature_type
            rng = self._chain._rng
            self._states = self._chain._in_window_states if self._state_type == IN_WINDOW else self._chain._out_window_states
            n_states = len(self._states)
            self._p = rng.uniform(size=n_states)
            if feature_type == CONTINUOUS:
                mean = rng.uniform(0.1)
                sd = rng.uniform(0.1) * 0.05
                if self._chain._trends:
                    if self._index == 0:
                        pass  # Increase
                    elif self._index == 1:
                        mean = -mean  # Decrease
                    elif self._index == 2:
                        mean = 0  # Stay constant
                    else:
                        mean = rng.uniform(-1, 1)  # Random
                self.sample = lambda: rng.normal(mean, sd)
            elif feature_type in {BINARY, CATEGORICAL, ORDINAL}:
                self.sample = lambda: self._index
                if feature_type == ORDINAL:
                    # Dissallow transitions to non-consecutive states:
                    mask = np.zeros(n_states)
                    mask[max(0, self._index - 1): min(n_states, self._index + 2)] = 1
                    self._p *= mask
            else:
                raise ValueError("Feature type invalid: {0}".format(feature_type))
            self._p /= self._p.sum()  # normalize transition probabilities

        def transition(self):
            """Transition to next state"""
            assert self._states is not None, "gen_distributions must be invoked first"
            return self._chain._rng.choice(self._states, p=self._p)

    def __init__(self, rng, feature_type, window, **kwargs):
        super().__init__(rng, feature_type, window)
        n_states = 2 if self._feature_type == BINARY else kwargs.get("n_states", self._rng.integers(2, 5, endpoint=True))
        self._window_independent = kwargs.get("window_independent", True)
        self._trends = False  # If enabled, sampled values increase/decrease/stay constant according to trends corresponding to each state
        self._init_value = self._rng.uniform(-1, 1)  # Initial value of Markov chain, used for trends
        if self._feature_type == CONTINUOUS:
            self._trends = self._rng.choice([True, False])
        # Select states inside and outside window
        state_idx = set(range(n_states))
        if self._window_independent:
            states = [self.State(self, index, IN_WINDOW) for index in state_idx]
            self._in_window_states = states
            self._out_window_states = states
            for state in states:
                state.gen_distributions()
        else:
            num_window_states = self._rng.integers(1, n_states)  # up to one less than the total number of states
            in_window_states_idx = self._rng.choice(range(n_states), size=num_window_states, replace=False)
            out_window_states_idx = state_idx.difference(in_window_states_idx)
            self._in_window_states = [self.State(self, index, IN_WINDOW) for index in in_window_states_idx]
            self._out_window_states = [self.State(self, index, OUT_WINDOW) for index in out_window_states_idx]
            for state in self._in_window_states + self._out_window_states:
                state.gen_distributions()

    def sample(self, sequence_length):
        cur_state = self._rng.choice(self._out_window_states)  # initial state
        sequence = np.empty(sequence_length)
        value = self._init_value  # TODO: what if value is re-initialized for every sequence sampled? (trends)
        left, right = self._window
        for timestep in range(sequence_length):
            if not self._window_independent:
                # Reset initial state in/out of window
                if timestep == left:
                    cur_state = self._rng.choice(self._in_window_states)
                elif timestep == right:
                    cur_state = self._rng.choice(self._out_window_states)
            # Get value from state
            if self._trends:
                value += cur_state.sample()
            else:
                value = cur_state.sample()
            sequence[timestep] = value
            # Set next state
            cur_state = cur_state.transition()
        return sequence
