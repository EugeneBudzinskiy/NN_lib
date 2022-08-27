from abc import ABC
from abc import abstractmethod


class AbstractVariable(ABC):
    def __init__(self, value: float, partial: float = 0.):
        self.value = value
        self.partial = partial

    def _wrapper(self, other):
        return other if isinstance(other, self.__class__) else self.__class__(other)

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __sub__(self, other):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __truediv__(self, other):
        pass

    @abstractmethod
    def __pow__(self, power, modulo=None):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __ne__(self, other):
        pass

    @abstractmethod
    def __le__(self, other):
        pass

    @abstractmethod
    def __ge__(self, other):
        pass

    @abstractmethod
    def __lt__(self, other):
        pass

    @abstractmethod
    def __gt__(self, other):
        pass

    @abstractmethod
    def __neg__(self):
        pass

    @abstractmethod
    def __pos__(self):
        pass

    @abstractmethod
    def sqrt(self):
        pass

    @abstractmethod
    def exp(self):
        pass

    @abstractmethod
    def log(self):
        pass

    @abstractmethod
    def log2(self):
        pass

    @abstractmethod
    def log10(self):
        pass

    @abstractmethod
    def sin(self):
        pass

    @abstractmethod
    def cos(self):
        pass

    @abstractmethod
    def tan(self):
        pass

    @abstractmethod
    def arcsin(self):
        pass

    @abstractmethod
    def arccos(self):
        pass

    @abstractmethod
    def arctan(self):
        pass

    @abstractmethod
    def sinh(self):
        pass

    @abstractmethod
    def cosh(self):
        pass

    @abstractmethod
    def tanh(self):
        pass


class AbstractOperation(ABC):
    def __init__(self):
        self.epsilon = 1e-7

    @abstractmethod
    def __call__(self, *args, **kwargs) -> AbstractVariable:
        pass


class UniOperation(AbstractOperation):
    @abstractmethod
    def __call__(self, x: AbstractVariable) -> AbstractVariable:
        pass


class BiOperation(AbstractOperation):
    @abstractmethod
    def __call__(self, x1: AbstractVariable, x2: AbstractVariable) -> AbstractVariable:
        pass