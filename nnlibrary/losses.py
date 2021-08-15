from nnlibrary.errors import WrongLoss

from numpy import mean
from numpy import square


def mse(y_predicted, y_target):
    return mean(square(y_target - y_predicted))


loss_dict = {
    'mse': mse
}


def get_loss(name: str):
    if type(name) == str:
        low_name = name.lower()
        return loss_dict[low_name] if low_name in loss_dict else None
    else:
        raise ValueError(f'Loss function name should be `str` type, instead has `{type(name).__name__}` type')

