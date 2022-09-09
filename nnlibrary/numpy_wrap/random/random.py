from typing import Any
from typing import Union
from typing import Optional
from typing import Iterable

import nnlibrary.numpy_wrap as npw

from nnlibrary.numpy_wrap.node import AbstractNode
from nnlibrary.numpy_wrap.node import Node
from nnlibrary.numpy_wrap.node import node_utils


def beta(a: Union[float, npw.ndarray, Iterable, int, AbstractNode],
         b: Union[float, npw.ndarray, Iterable, int, AbstractNode],
         size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    a = node_utils.convert_to_node_if_needed(x=a)
    b = node_utils.convert_to_node_if_needed(x=b)
    values = npw.numpy.random.beta(a=a.values, b=b.values, size=size)
    return Node(values=values)  # partials would be zeros


def binomial(n: Union[int, npw.ndarray, Iterable, float, AbstractNode],
             p: Union[float, npw.ndarray, Iterable, int, AbstractNode],
             size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    n = node_utils.convert_to_node_if_needed(x=n)
    p = node_utils.convert_to_node_if_needed(x=p)
    values = npw.numpy.random.binomial(n=n.values, p=p.values, size=size)
    return Node(values=values)  # partials would be zeros


def chisquare(df: Union[float, npw.ndarray, Iterable, int, AbstractNode],
              size: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    df = node_utils.convert_to_node_if_needed(x=df)
    values = npw.numpy.random.chisquare(df=df.values, size=size)
    return Node(values=values)  # partials would be zeros


def choice(a: Union[Any, AbstractNode],
           size: Union[int, Iterable, tuple[int], None] = None,
           replace: Optional[bool] = True,
           p: Union[Any, AbstractNode] = None) -> AbstractNode:
    p = node_utils.get_values_if_needed(x=p)
    a = node_utils.convert_to_node_if_needed(x=a)
    idx = npw.numpy.random.choice(a=a.values.size, size=size, replace=replace, p=p)
    return Node(values=a.values[idx], partials=a.partials[idx])
