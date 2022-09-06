from typing import Any
from typing import Union
from typing import Optional
from typing import Iterable

import nnlibrary.numpy_wrap as npw

from nnlibrary.numpy_wrap.node import AbstractNode
from nnlibrary.numpy_wrap.node import Node
from nnlibrary.numpy_wrap.node import node_utils


def sin(x: Union[npw.ndarray, Iterable, AbstractNode],
        *args: Any,
        **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.sin(x=x.values, *args, **kwargs)
    partials = x.partials * npw.numpy.cos(x=x.values)
    return Node(values=values, partials=partials)


def cos(x: Union[npw.ndarray, Iterable, AbstractNode],
        *args: Any,
        **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.cos(x=x.values, *args, **kwargs)
    partials = - x.partials * npw.numpy.sin(x=x.values)
    return Node(values=values, partials=partials)


def tan(x: Union[npw.ndarray, Iterable, AbstractNode],
        *args: Any,
        **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.tan(x=x.values, *args, **kwargs)
    partials = x.partials / npw.numpy.cos(x.values) ** 2
    return Node(values=values, partials=partials)


def arcsin(x: Union[npw.ndarray, Iterable, AbstractNode],
           *args: Any,
           **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.arcsin(x=x.values, *args, **kwargs)
    partials = x.partials / npw.numpy.sqrt(1 - x.values ** 2)
    return Node(values=values, partials=partials)


def arccos(x: Union[npw.ndarray, Iterable, AbstractNode],
           *args: Any,
           **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.arccos(x=x.values, *args, **kwargs)
    partials = - x.partials / npw.numpy.sqrt(1 - x.values ** 2)
    return Node(values=values, partials=partials)


def arctan(x: Union[npw.ndarray, Iterable, AbstractNode],
           *args: Any,
           **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.arctan(x=x.values, *args, **kwargs)
    partials = x.partials / (1 + x.values ** 2)
    return Node(values=values, partials=partials)


def sinh(x: Union[npw.ndarray, Iterable, AbstractNode],
         *args: Any,
         **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.sinh(x=x.values, *args, **kwargs)
    partials = x.partials * npw.numpy.cosh(x.values)
    return Node(values=values, partials=partials)


def cosh(x: Union[npw.ndarray, Iterable, AbstractNode],
         *args: Any,
         **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.cosh(x=x.values, *args, **kwargs)
    partials = x.partials * npw.numpy.sinh(x.values)
    return Node(values=values, partials=partials)


def tanh(x: Union[npw.ndarray, Iterable, AbstractNode],
         *args: Any,
         **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.tanh(x=x.values, *args, **kwargs)
    partials = x.partials * (1 - values ** 2)
    return Node(values=values, partials=partials)


def arcsinh(x: Union[npw.ndarray, Iterable, AbstractNode],
            *args: Any,
            **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.arcsinh(x=x.values, *args, **kwargs)
    partials = x.partials * npw.numpy.sqrt(x.values ** 2 + 1)
    return Node(values=values, partials=partials)


def arccosh(x: Union[npw.ndarray, Iterable, AbstractNode],
            *args: Any,
            **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.arccosh(x=x.values, *args, **kwargs)
    partials = x.partials * npw.numpy.sqrt(x.values ** 2 - 1)
    return Node(values=values, partials=partials)


def arctanh(x: Union[npw.ndarray, Iterable, AbstractNode],
            *args: Any,
            **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.arctanh(x=x.values, *args, **kwargs)
    partials = x.partials * (1 - x.values ** 2)
    return Node(values=values, partials=partials)
