from .abstractions import AbstractNode


class Node(AbstractNode):
    def __repr__(self):
        return f'{self.values}'

    @property
    def shape(self) -> tuple:
        assert self.values.shape == self.partials.shape, 'Incoherent shapes!'
        return self.values.shape

