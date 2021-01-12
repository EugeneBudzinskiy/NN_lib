class InputLayerNotDefined(Exception):
    def __init__(self):
        self.text = 'Input layer is not defined'
        super(InputLayerNotDefined, self).__init__(self.text)


class WrongStructure(Exception):
    def __init__(self):
        self.text = 'Neural Network has wrong structure'
        super(WrongStructure, self).__init__(self.text)