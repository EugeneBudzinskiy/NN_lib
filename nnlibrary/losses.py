from numpy import square
from numpy import sum

from nnlibrary import errors


def mse(y_predicted, y_target, for_training=False):
    loss = square(y_target - y_predicted) / y_target.shape[-1]
    return loss if for_training else sum(loss)


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

