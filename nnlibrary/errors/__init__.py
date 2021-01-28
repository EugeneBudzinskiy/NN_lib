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


class IsNotALayer(Exception):
    def __init__(self, variable_name):
        self.text = f'`{type(variable_name)}` is not Layer type'
        super(IsNotALayer, self).__init__(self.text)

