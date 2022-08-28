from nnlibrary.auto_diff.forward_mode import ForwardMode
from nnlibrary.auto_diff.reverse_mode import ReverseMode


class AutoDiff:
    forward_mode = ForwardMode()
    backward_mode = ReverseMode()
