from typing import Any
from typing import Union
from typing import Optional
from typing import Iterable

import nnlibrary.numpy_wrap as npw

from nnlibrary.numpy_wrap.node import AbstractNode
from nnlibrary.numpy_wrap.node import Node
from nnlibrary.numpy_wrap.node import node_utils


def dot(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
        b: Union[npw.ndarray, Iterable, int, float, AbstractNode],
        out: Optional[Union[npw.ndarray, AbstractNode]] = None) -> AbstractNode:
    a = node_utils.convert_to_node_if_needed(x=a)
    b = node_utils.convert_to_node_if_needed(x=b)

    partials = npw.numpy.dot(a=a.partials, b=b.values) + npw.numpy.dot(a=a.values, b=b.partials)

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = npw.numpy.dot(a=a.values, b=b.values, out=out.values), partials
        return out

    values = npw.numpy.dot(a=a.values, b=b.values, out=out)
    return Node(values=values, partials=partials)


def multi_dot(arrays: Union[Iterable[npw.ndarray], Iterable, int, float, Iterable[AbstractNode]],
              out: Optional[Union[npw.ndarray, AbstractNode]] = None) -> AbstractNode:

    result = npw.ones(1)
    for arr in arrays:
        result = npw.dot(result, arr)

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = result.values, result.partials
        return out

    return result
