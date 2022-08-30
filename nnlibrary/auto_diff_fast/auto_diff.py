from nnlibrary.auto_diff_fast.forward_mode import ForwardMode
from nnlibrary.auto_diff_fast.reverse_mode import ReverseMode


class AutoDiff:
    forward_mode = ForwardMode()
    backward_mode = ReverseMode()
