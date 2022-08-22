import numpy as np

from nnlibrary.reductions import AbstractReduction


class ReductionNone(AbstractReduction):
    def __call__(self, values: np.ndarray) -> np.ndarray:
        return values.copy()


class ReductionSum(AbstractReduction):
    def __call__(self, values: np.ndarray) -> np.ndarray:
        return np.sum(values, axis=-1)


class ReductionMean(AbstractReduction):
    def __call__(self, values: np.ndarray) -> np.ndarray:
        return np.mean(values, axis=-1)
