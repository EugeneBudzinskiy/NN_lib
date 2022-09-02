from typing import Iterable
from typing import Union

import numpy as np

from nnlibrary.nodes import Node


def empty(shape: Union[int, Iterable, tuple[int]], *args, **kwargs) -> Node:
    return Node(values=np.empty(shape=shape, *args, **kwargs),
                partials=np.zeros(shape=shape, *args, **kwargs))


def empty_like(prototype: Union[np.ndarray, Iterable, int, float], *args, **kwargs) -> Node:
    return Node(values=np.empty_like(prototype=prototype, *args, **kwargs),
                partials=np.zeros_like(prototype=prototype, *args, **kwargs))


