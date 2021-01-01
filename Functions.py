class Func:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self):
        self.__functions = {
            'linear': (
                self.linear,
                self.linear_der
            ),
            'relu': (
                self.relu,
                self.relu_der
            ),
            'sigmoid': (
                self.sigmoid,
                self.sigmoid_der
            )
        }

    def get(self, func_name: str):
        if func_name in self.__functions:
            return self.__functions[func_name]

        raise NotImplementedError(
            f"The activation function '{func_name}' doesn't exist"  # TODO MB Rewrite Error
        )

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_der(x):
        return x

    @staticmethod
    def relu(x):
        return x

    @staticmethod
    def relu_der(x):
        return x

    @staticmethod
    def sigmoid(x):
        return x

    @staticmethod
    def sigmoid_der(x):
        return x
