"""Generator base class"""

from abc import ABC
from collections import namedtuple

import numpy as np
import graphviz
from scipy.stats import bernoulli

from synmod.constants import NUMERIC

IN_WINDOW = "in-window"
OUT_WINDOW = "out-window"
SummaryStats = namedtuple("SummaryStats", ["mean", "sd"])


class TabularGenerator():
    """Tabular feature generator"""
    def __init__(self, rng):
        self._rng = rng

    def sample(self):
        """Sample i.i.d. from generator"""

    def summary(self):
        """Summary of generator parameters"""


class BernoulliDistribution(TabularGenerator):
    """Bernoulli distribution generator"""
    def __init__(self, rng, **kwargs):
        super().__init__(rng)
        self._p = kwargs.get("p", self._rng.uniform(0.01, 0.99))

    def sample(self):
        return self._rng.binomial(1, self._p)

    def summary(self):
        return dict(name=self.__class__.__name__,
                    prob=self._p)


class CategoricalDistribution(TabularGenerator):
    """Categorical distribution generator"""
    def __init__(self, rng, **kwargs):
        super().__init__(rng)
        self._size = kwargs.get("size", self._rng.integers(2, 5, endpoint=True))
        self._values = np.arange(2, 2 + self._size)
        self._p = rng.uniform(size=self._size)
        self._p /= self._p.sum()

    def sample(self):
        return self._rng.choice(self._values, p=self._p)

    def summary(self):
        return dict(name=self.__class__.__name__,
                    values=self._values,
                    probs=self._p)


class NormalDistribution(TabularGenerator):
    """Normal distribution generator"""
    def __init__(self, rng):
        super().__init__(rng)
        self._mean = rng.uniform(-1, 1)
        self._sd = rng.uniform(0.1) * 0.05

    def sample(self):
        return self._rng.normal(self._mean, self._sd)

    def summary(self):
        return dict(name=self.__class__.__name__,
                    mean=self._mean,
                    sd=self._sd)


class Generator(ABC):
    """Sequence generator base class"""
    def __init__(self, rng, feature_type, window):
        self._rng = rng
        self._feature_type = feature_type
        self._window = window

    def sample(self, sequence_length):
        """Sample sequence of given length from generator"""

    def graph(self):
        """Graph representation of generator (dot file)"""

    def summary(self):
        """Summary of generator"""


class BernoulliProcess(Generator):
    """Bernoulli process generator"""
    def __init__(self, rng, feature_type, window, **kwargs):
        super().__init__(rng, feature_type, window)
        self._p = kwargs.get("p", self._rng.uniform(0.01, 0.99))
        self._window_independent = kwargs.get("window_independent", True)  # Sampled value independent of window location

    def sample(self, sequence_length):
        sequence = bernoulli.rvs(p=self._p, size=sequence_length, random_state=self._rng)
        if not self._window_independent:
            left, right = self._window
            sequence[left: right + 1] = bernoulli.rvs(p=(1 - self._p), size=(right - left + 1), random_state=self._rng)
        return sequence

    def graph(self):
        graph = graphviz.Digraph()
        left, right = self._window
        graph.attr(label=f"Bernoulli process\nWindow: [{left}, {right}]\n\n", labelloc="t")
        cnames = ["0"]
        cprobs = [self._p]
        clabels = [""]
        if not self._window_independent:
            cnames.append("1")
            cprobs.append(1 - self._p)
            clabels = ["Out-of-window state", "In-window state"]
        for cidx, cname in enumerate(cnames):
            with graph.subgraph(name=f"cluster_{cidx}") as cgraph:
                cgraph.attr(label=clabels[cidx])
                cgraph.node(cname, label=f"P(X = 1) = {cprobs[cidx]:1.5f}")
                cgraph.edge(cname, cname, " 1.0")
        return graph

    def summary(self):
        in_window_prob = self._p if self._window_independent else 1 - self._p
        return dict(out_window_prob=self._p, in_window_prob=in_window_prob)


# pylint: disable = invalid-name, pointless-statement
class MarkovChain(Generator):
    """Markov chain generator"""

    class State():
        """Markov chain state"""
        # pylint: disable = protected-access, too-many-instance-attributes
        def __init__(self, chain, index, state_type):
            self._chain = chain  # Parent Markov chain
            self._index = index  # state identifier
            self._state_type = state_type  # state type - in-window vs. out-window
            self.name = f"{self._state_type}-{self._index}"
            self._p = None  # Transition probabilities from state
            self._states = None  # States to transition to
            self.sample = None  # Function to sample from state distribution
            if self._chain._feature_type == NUMERIC:
                self._summary_stats = SummaryStats(None, None)

        def gen_distributions(self):
            """Generate state transition and sampling distributions"""
            feature_type = self._chain._feature_type
            rng = self._chain._rng
            self._states = self._chain._in_window_states if self._state_type == IN_WINDOW else self._chain._out_window_states
            n_states = len(self._states)
            self._p = rng.uniform(size=n_states)
            if feature_type == NUMERIC:
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
                self._summary_stats = SummaryStats(mean, sd)
                self.sample = lambda: rng.normal(mean, sd)
            else:  # binary/categorical variable
                self.sample = lambda: self._index
            self._p /= self._p.sum()  # normalize transition probabilities

        def transition(self):
            """Transition to next state"""
            assert self._states is not None, "gen_distributions must be invoked first"
            return self._chain._rng.choice(self._states, p=self._p)

    def __init__(self, rng, feature_type, window, **kwargs):
        super().__init__(rng, feature_type, window)
        n_states = kwargs.get("n_states", self._rng.integers(2, 5, endpoint=True))
        self._window_independent = kwargs.get("window_independent", False)  # Sampled state independent of window location
        # If trends enabled, sampled values increase/decrease/stay constant according to trends corresponding to each state:
        self._trends = self._rng.choice([True, False]) if self._feature_type == NUMERIC else False
        if self._trends and not self._window_independent:
            n_states = min(n_states, 4)  # Separate chains in/out of window, so avoid too many trends within window
        self._init_value = self._rng.uniform(-1, 1)  # Initial value of Markov chain, used for trends
        # Select states inside and outside window
        self._in_window_states = [self.State(self, index, IN_WINDOW) for index in range(n_states)]
        self._out_window_states = self._in_window_states
        if not self._window_independent:
            # Create separate chain in/out of window
            self._out_window_states = [self.State(self, index, OUT_WINDOW) for index in range(n_states)]
        states = self._in_window_states if self._window_independent else self._in_window_states + self._out_window_states
        for state in states:
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
                elif timestep == right + 1:
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

    def graph(self):
        graph = graphviz.Digraph()
        label = f"Markov chain\nFeature type: {self._feature_type}"
        if self._trends:
            label += f"\nTrends: True\nInitial value: {self._init_value:1.5f}"
        left, right = self._window
        label += f"\nWindow: [{left}, {right}]\n\n"
        graph.attr(label=label, labelloc="t")
        clusters = [self._in_window_states]
        clabels = [""]
        if not self._window_independent:
            clusters.append(self._out_window_states)
            clabels = ["In-window states", "Out-of-window states"]
        for cidx, cluster in enumerate(clusters):
            with graph.subgraph(name=f"cluster_{cidx}") as cgraph:
                cgraph.attr(label=clabels[cidx])
                for state in cluster:
                    # pylint: disable = protected-access
                    label = f"State {state._index}"
                    if self._feature_type == NUMERIC:
                        label += f"\nMean: {state._summary_stats.mean:1.5f}\nSD: {state._summary_stats.sd:1.5f}"
                    cgraph.node(state.name, label=label)
                    for oidx, ostate in enumerate(cluster):
                        cgraph.edge(state.name, ostate.name, label=f" {state._p[oidx]:1.5f}\t\n")
        return graph

    def summary(self):
        summary = {}
        if self._feature_type == NUMERIC:
            summary["trends"] = self._trends
            if self._trends:
                summary["init_value"] = self._init_value
        for stype, states in {"out_window_states": self._out_window_states, "in_window_states": self._in_window_states}.items():
            states_summary = [None] * len(states)
            for idx, state in enumerate(states):
                state_summary = {}
                # pylint: disable = protected-access
                state_summary["index"] = state._index
                state_summary["p"] = state._p
                if self._feature_type == NUMERIC:
                    mean, sd = state._summary_stats
                    state_summary["stats"] = dict(mean=mean, sd=sd)
                states_summary[idx] = state_summary
            summary[stype] = states_summary
        return summary
