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


def inner(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
          b: Union[npw.ndarray, Iterable, int, float, AbstractNode]) -> AbstractNode:
    a = node_utils.convert_to_node_if_needed(x=a)
    b = node_utils.convert_to_node_if_needed(x=b)

    values = npw.numpy.inner(a=a.values, b=b.values)
    partials = npw.numpy.inner(a=a.partials, b=b.values) + npw.numpy.inner(a=a.values, b=b.partials)
    return Node(values=values, partials=partials)


def outer(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
          b: Union[npw.ndarray, Iterable, int, float, AbstractNode],
          out: Optional[Union[npw.ndarray, AbstractNode]] = None) -> AbstractNode:
    a = node_utils.convert_to_node_if_needed(x=a)
    b = node_utils.convert_to_node_if_needed(x=b)

    partials = npw.numpy.outer(a=a.partials, b=b.values) + npw.numpy.outer(a=a.values, b=b.partials)

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = npw.numpy.outer(a=a.values, b=b.values, out=out.values), partials
        return out

    values = npw.numpy.outer(a=a.values, b=b.values)
    return Node(values=values, partials=partials)


def matmul(x1: Union[npw.ndarray, Iterable, int, float, AbstractNode],
           x2: Union[npw.ndarray, Iterable, int, float, AbstractNode],
           *args: Any,
           **kwargs: Any) -> AbstractNode:
    x1 = node_utils.convert_to_node_if_needed(x=x1)
    x2 = node_utils.convert_to_node_if_needed(x=x2)

    values = npw.numpy.matmul(x1=x1.values, x2=x2.values, *args, **kwargs)
    partials = npw.numpy.matmul(x1=x1.partials, x2=x2.values) + npw.numpy.matmul(x1=x1.values, x2=x2.partials)

    return Node(values=values, partials=partials)


def matrix_power(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
                 n: int) -> AbstractNode:
    result = npw.copy(a=a)
    for _ in range(n - 1):
        result = npw.matmul(x1=result, x2=a)

    return result
