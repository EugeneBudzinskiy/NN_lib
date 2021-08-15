from numpy import mean
from numpy import square

from nnlibrary import errors


def mse(y_predicted, y_target):
    return mean(square(y_target - y_predicted))


loss_dict = {
    'mse': mse
}


def get_loss(name):
    if callable(name):
        return name
    elif type(name) == str:
        low_name = name.lower()
        if low_name in loss_dict:
            return loss_dict[low_name]
        else:
            raise errors.WrongLoss(name)
    else:
        raise ValueError(f'Loss function name should be `str` type, instead has `{type(name).__name__}` type')

