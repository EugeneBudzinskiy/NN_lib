class InputLayerNotDefined(Exception):
    def __init__(self):
        self.text = 'Input layer is not defined'
        super(InputLayerNotDefined, self).__init__(self.text)


class InputLayerAlreadyDefined(Exception):
    def __init__(self):
        self.text = 'Input layer is already defined'
        super(InputLayerAlreadyDefined, self).__init__(self.text)


class WrongStructure(Exception):
    def __init__(self):
        self.text = 'Model has wrong structure'
        super(WrongStructure, self).__init__(self.text)


class NothingToPop(Exception):
    def __init__(self):
        self.text = 'Model has no layer`s to pop'
        super(NothingToPop, self).__init__(self.text)


class AlreadyCompiled(Exception):
    def __init__(self):
        self.text = 'Model was already been compiled'
        super(AlreadyCompiled, self).__init__(self.text)


class NotCompiled(Exception):
    def __init__(self):
        self.text = 'Model has`t been compiled'
        super(NotCompiled, self).__init__(self.text)


class TryModifyCompiledNN(Exception):
    def __init__(self):
        self.text = 'Try to modify already compiled Model'


class WrongLayerIndex(Exception):
    def __init__(self):
        self.text = 'Given layer index are wrong'
        super(WrongLayerIndex, self).__init__(self.text)


class ProvideLayerIndex(Exception):
    def __init__(self):
        self.text = 'Provide the layer index'
        super(ProvideLayerIndex, self).__init__(self.text)


class WrongStructureElement(Exception):
    def __init__(self, variable_name):
        self.text = f'Element of the structure `{type(variable_name)}` is not Layer type'
        super(WrongStructureElement, self).__init__(self.text)


class IsNotALayer(Exception):
    def __init__(self, variable_name):
        self.text = f'`{variable_name}` need to be `Layer` type, but instead is `{type(variable_name).__name__}` type'
        super(IsNotALayer, self).__init__(self.text)


class WrongInputShape(Exception):
    def __init__(self, shape_a, shape_b):
        self.text = f'Input data should have shape {shape_a}, but instead it has shape {shape_b}'
        super(WrongInputShape, self).__init__(self.text)


class OptimizerNotSpecify(Exception):
    def __init__(self):
        self.text = f'Neural Network should has Optimizer'
        super(OptimizerNotSpecify, self).__init__(self.text)


class LossNotSpecify(Exception):
    def __init__(self):
        self.text = f'Neural Network should has Loss function'
        super(LossNotSpecify, self).__init__(self.text)


class WrongOptimizer(Exception):
    def __init__(self, name):
        self.text = f'Optimizer with name `{name}` was not found in library'
        super(WrongOptimizer, self).__init__(self.text)


class WrongLoss(Exception):
    def __init__(self, name):
        self.text = f'Loss function with name `{name}` was not found in library'
        super(WrongLoss, self).__init__(self.text)
