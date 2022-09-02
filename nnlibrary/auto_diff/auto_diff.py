from .forward_mode import ForwardMode
from .reverse_mode import ReverseMode


class AutoDiff:
    forward_mode = ForwardMode()
    backward_mode = ReverseMode()
