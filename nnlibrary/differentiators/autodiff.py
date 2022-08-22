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

    def __mul__(self, other):
        result = Variable(value=self.value * other.value)
        result.dot = self.dot * other.value + self.value * other.dot
        return result

    def __truediv__(self, other):
        result = Variable(value=self.value / other.value)
        result.dot = (self.dot * other.value - self.value * other.dot) / other.value ** 2
        return result


class AutomaticDifferentiation:
    pass
