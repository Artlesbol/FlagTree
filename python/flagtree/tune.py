# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from functools import cached_property, partial, update_wrapper
from math import isfinite
from numbers import Real
import warnings

__all__ = ["AutoTuner", "autotune", "to_hashable"]


def to_hashable(tensor):
    """Return a tensor's shape, strides and dtype as a hashable workload feature."""
    return tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype


class AutoTuner:

    def __init__(self, func, config_space, key_fn: Callable, prune_fn: Callable | None = None, *, do_bench=None):
        self.config_space = config_space
        if not self.config_space:
            raise ValueError("config_space must contain at least one configuration")
        if any(not isinstance(config, dict) for config in self.config_space):
            raise TypeError("Each configuration must be a dictionary of function keyword arguments")
        self.fn = func
        self.key_fn = key_fn
        self.prune_fn = prune_fn
        self.best_configs = {}
        self._do_bench = do_bench
        update_wrapper(self, func, updated=())

    @cached_property
    def do_bench(self):
        if self._do_bench is not None:
            return self._do_bench
        # Resolve the active backend only when benchmarking is actually needed.
        from triton.runtime.driver import driver

        return driver.active.get_benchmarker()

    def get_pruned_config(self, *args, **kwargs):
        if self.prune_fn is None:
            return self.config_space
        return [config for config in self.config_space if self.prune_fn(config, *args, **kwargs)]

    def __call__(self, *args, autotune=True, autotune_allow_arg_overwrite=False, **kwargs):
        tuned_names = {name for name in self.config_space[0]}
        conflicts = tuned_names.intersection(kwargs)
        if conflicts:
            if not autotune_allow_arg_overwrite:
                raise ValueError(f"Arguments are also supplied by config_space: {', '.join(sorted(conflicts))}")
            else:
                warnings.warn(
                    "Duplicate keyword arguments, but autotune_allow_arg_overwrite is True, will use the argument provided by the user."
                )

        if not autotune:
            return self.fn(*args, **{**self.config_space[0], **kwargs})

        # Only workload arguments reach key_fn; tunable arguments are injected later.
        key = self.key_fn(*args, **kwargs) if self.key_fn is not None else None
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError("key_fn must return a hashable value") from exc

        best_config = self.best_configs.get(key)
        if best_config is None:
            config_space = self.get_pruned_config(*args, **kwargs)
            if not config_space:
                raise ValueError("prune_fn removed all configurations")
            timings = self.tune(config_space, *args, **kwargs)
            best_config = timings[0][1]
            self.best_configs[key] = best_config

        # Benchmark results are timings, so execute the winner to obtain the real result.
        return self.fn(*args, **{**best_config, **kwargs})

    def tune(self, config_space, *args, **kwargs):
        """Return (latency_ms, config) pairs sorted from fastest to slowest.

        As in triton_dist, a candidate that raises during benchmarking is assigned
        infinite latency. If every candidate fails, raise instead of caching an
        unusable configuration. Backend initialization errors propagate directly.
        """
        benchmark = self.do_bench
        timings = []
        first_error = None
        for config in config_space:
            bench_func = partial(self.fn, *args, **{**config, **kwargs})
            try:
                # The backend benchmarker owns warmup, repetition and synchronization.
                measurement = benchmark(bench_func, quantiles=[0.5])
                # Triton's default benchmarker unwraps a single quantile to a scalar.
                latency = float(measurement if isinstance(measurement, Real) else measurement[0])
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                latency = float("inf")
            if not isfinite(latency) or latency < 0:
                latency = float("inf")
            timings.append((latency, config))

        timings.sort(key=lambda item: item[0])
        if not timings or not isfinite(timings[0][0]):
            raise RuntimeError("No configuration produced a valid benchmark result") from first_error
        return timings


def autotune(config_space, key_fn, prune_fn=None, *, do_bench=None):
    """Tune a Python function over candidate keyword-argument dictionaries.

    ``config_space`` contains dictionaries such as ``{"config": triton.Config(...)}``.
    Values are passed unchanged to the decorated function. ``key_fn(*args, **kwargs)``
    returns a hashable workload key; use ``None`` to share one cache entry for all
    inputs. Include every input feature that affects configuration validity or
    performance in this key. ``prune_fn(config, *args, **kwargs)`` returns True to
    keep a candidate and False to discard it, and only runs on a cache miss.

    ``do_bench(fn, quantiles=[0.5])`` returns a median latency in milliseconds, either
    as a scalar or a one-element sequence. It defaults to the active FlagTree
    backend's benchmarker. Tuning executes the function repeatedly; inputs must
    tolerate these repeated calls.

    Call with ``autotune=False`` to use the first unpruned candidate directly.
    Candidate names must not also be supplied by the caller unless
    ``autotune_allow_arg_overwrite=True``, in which case caller keywords take
    precedence. These two control keywords are reserved by the wrapper.
    ``decorated.fn`` exposes the original function, and
    ``decorated.best_configs.clear()`` clears the in-memory cache.

    Example::

        @autotune(
            config_space=[{"block_size": 32}, {"block_size": 64}],
            key_fn=lambda x: to_hashable(x),
            prune_fn=lambda config, x: config["block_size"] <= x.numel(),
        )
        def operation(x, block_size):
            ...  # Launch the kernel(s) using block_size.
    """

    def wrapper(func):
        return AutoTuner(func, config_space, key_fn, prune_fn, do_bench=do_bench)

    return wrapper
