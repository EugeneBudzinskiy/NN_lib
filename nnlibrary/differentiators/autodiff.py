from abc import ABC
from abc import abstractmethod


class Graph:
    def __init__(self):
        self.variables = list()

    def __enter__(self):
        Variable.graph = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        Variable.count = 0


class Node(ABC):
    def __init__(self, value, dot):
        self.value = value
        self.dot = dot

    @abstractmethod
    def __repr__(self):
        pass

    def __add__(self, other):
        result = Variable(value=self.value + other.value)
        result.dot = self.dot + other.dot
        return result

    def __sub__(self, other):
        result = Variable(value=self.value - other.value)
        result.dot = self.dot - other.dot
        return result

    def __neg__(self):
        result = Variable(value=- self.value)
        result.dot = - self.dot
        return result

    def __pos__(self):
        result = Variable(value=self.value)
        result.dot = self.dot
        return result

    def __mul__(self, other):
        result = Variable(value=self.value * other.value)
        result.dot = self.dot * other.value + self.value * other.dot
        return result

    def __truediv__(self, other):
        result = Variable(value=self.value / other.value)
        result.dot = (self.dot * other.value - self.value * other.dot) / other.value ** 2
        return result

    def __pow__(self, power, modulo=None):
        result = Variable(value=self.value ** power.value)
        result.dot = self.dot * power.value * self.value ** (power.value - 1)
        return result


class Variable(Node):
    count = 0
    graph = Graph()

    def __init__(self, value, dot=0., name=None):
        super(Variable, self).__init__(value=value, dot=dot)
        Variable.graph.variables.append(self)
        self.name = f"Var/{Variable.count}" if name is None else name
        Variable.count += 1

    def __repr__(self):
        return f"Variable: name:{self.name}, value:{self.value}"


class AutomaticDifferentiation:
    pass
