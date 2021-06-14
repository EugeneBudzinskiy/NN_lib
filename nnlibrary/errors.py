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
        self.text = 'Neural Network has wrong structure'
        super(WrongStructure, self).__init__(self.text)


class NothingToPop(Exception):
    def __init__(self):
        self.text = 'Neural Network has no layer`s to pop'
        super(NothingToPop, self).__init__(self.text)


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
