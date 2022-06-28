from abc import ABC
from abc import abstractmethod

from numpy import ndarray

from activation_fucntions import Exponent
from activation_fucntions import HardSigmoid
from activation_fucntions import Linear
from activation_fucntions import ReLU
from activation_fucntions import Sigmoid
from activation_fucntions import TanH


class Activation(ABC):
    @abstractmethod
    def __call__(self, x: ndarray):
        pass


class ActivationMapper:
    available_func = {
        'linear': Linear,
        'hard_sigmoid': HardSigmoid,
        'sigmoid': Sigmoid,
        'relu': ReLU,
        'exponent': Exponent,
        'tanh': TanH
    }

    @staticmethod
    def get_activation_function(name: str):
        return ActivationMapper.available_func[name]
