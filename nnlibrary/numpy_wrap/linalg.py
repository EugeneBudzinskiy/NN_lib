from typing import Any
from typing import Union
from typing import Optional
from typing import Iterable

import nnlibrary.numpy_wrap as npw

from .node import AbstractNode
from .node import Node
from .node import node_utils


def dot(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
        b: Union[npw.ndarray, Iterable, int, float, AbstractNode],
        out: Optional[npw.ndarray, AbstractNode] = None) -> AbstractNode:
    a = node_utils.convert_to_node_if_needed(x=a)
    b = node_utils.convert_to_node_if_needed(x=b)
    out = node_utils.convert_to_node_if_needed(x=out)

    values = npw.numpy.dot(a=a.values, b=b.values, out=out.values)
    partials = npw.numpy.dot(a=a.partials, b=b.values) + npw.numpy.dot(a=a.values, b=b.partials)
    return Node(values=values, partials=partials)
