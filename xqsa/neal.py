"""
DWave neal simulated annealing backend.

Wraps dwave-neal's SimulatedAnnealingSampler to solve XQMX
quadratic models via simulated annealing on CPU.
"""

from __future__ import annotations

import time
from typing import Any

import dimod
import neal

from xqvm.core.xqmx import XQMX, XQMXDomain

from .backend import Backend, SolverResult

class NealBackend(Backend):
    """ Simulated annealing backend using dwave-neal. """

    def __init__(
        self,
        num_reads: int = 100,
        num_sweeps: int = 1000,
        beta_range: tuple[float, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.seed = seed

    def solve(self, model: XQMX, **kwargs: Any) -> SolverResult:
        """ Solve using simulated annealing via dwave-neal. """
        self._validate_model(model)

        num_reads = kwargs.get("num_reads", self.num_reads)
        num_sweeps = kwargs.get("num_sweeps", self.num_sweeps)
        beta_range = kwargs.get("beta_range", self.beta_range)
        seed = kwargs.get("seed", self.seed)

        bqm = self._model_to_bqm(model)
        sampler = neal.SimulatedAnnealingSampler()

        sample_kwargs: dict[str, Any] = {
            "num_reads": num_reads,
            "num_sweeps": num_sweeps,
        }
        if beta_range is not None:
            sample_kwargs["beta_range"] = beta_range
        if seed is not None:
            sample_kwargs["seed"] = seed

        t0 = time.perf_counter()
        result = sampler.sample(bqm, **sample_kwargs)
        elapsed = time.perf_counter() - t0

        best = result.first
        raw_sample = dict(best.sample)
        energy = float(best.energy)

        sample = self._sample_to_xqmx(model, raw_sample)

        return SolverResult(
            sample=sample,
            energy=energy,
            timing=elapsed,
            metadata={
                "num_reads": num_reads,
                "num_sweeps": num_sweeps,
                "beta_range": beta_range,
                "seed": seed,
                "num_occurrences": int(best.num_occurrences),
            },
        )

    def _model_to_bqm(self, model: XQMX) -> dimod.BinaryQuadraticModel:
        """ Convert an XQMX model to a dimod BQM. """
        vartype = (
            dimod.BINARY if model.domain == XQMXDomain.BINARY
            else dimod.SPIN
        )
        return dimod.BinaryQuadraticModel(
            model.linear,
            model.quadratic,
            0.0,
            vartype,
        )

    def _sample_to_xqmx(self, model: XQMX, raw_sample: dict[int, int]) -> XQMX:
        """ Convert a dimod sample dict to an XQMX sample. """
        if model.domain == XQMXDomain.BINARY:
            sample = XQMX.binary_sample(model.size, model.rows, model.cols)
        else:
            sample = XQMX.spin_sample(model.size, model.rows, model.cols)

        for var_idx, value in raw_sample.items():
            sample.set_linear(var_idx, float(value))

        return sample
