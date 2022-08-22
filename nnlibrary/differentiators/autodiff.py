# _g = None
#
#
# class Graph:
#     def __init__(self):
#         self.operators = set()
#         self.constants = set()
#         self.variables = set()
#         self.placeholders = set()
#         global _g
#         _g = self
#
#     def reset_counts(self, root):
#         if hasattr(root, 'count'):
#             root.count = 0
#         else:
#             for child in root.__subclasses__():
#                 self.reset_counts(child)
#
#     def reset_session(self):
#         try:
#             del _g
#         except NameError:
#             pass
#
#         self.reset_counts(Node)
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exc_type, exc_value, traceback):
#         self.reset_session()


class Variable:
    def __init__(self, value, dot=0.):
        self.value = value
        self.dot = dot

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


class AutomaticDifferentiation:
    pass
