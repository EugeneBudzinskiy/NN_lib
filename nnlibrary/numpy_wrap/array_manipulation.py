from typing import Any
from typing import Union
from typing import Optional
from typing import Iterable

import nnlibrary.numpy_wrap as npw

from .node import AbstractNode
from .node import Node
from .node import node_utils


def copyto(dst: Union[npw.ndarray, AbstractNode],
           src: Union[npw.ndarray, Iterable, int, float, AbstractNode],
           casting: Optional[str] = None,
           where: Union[npw.ndarray, Iterable, int, float, bool, None, AbstractNode] = None) -> None:
    dst = node_utils.get_values_if_needed(x=dst)
    src = node_utils.get_values_if_needed(x=src)
    where = node_utils.get_values_if_needed(x=where)
    npw.numpy.copyto(dst=dst, src=src, casting=casting, where=where)


def shape(a: Union[npw.ndarray, Iterable, int, float, AbstractNode]) -> tuple:
    a = node_utils.get_values_if_needed(x=a)
    return npw.numpy.shape(a)


def reshape(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
            newshape: Union[int, Iterable, tuple[int]],
            order: Optional[str] = 'C') -> AbstractNode:
    def wrap(x):
        return npw.numpy.reshape(a=x, newshape=newshape, order=order)

    if isinstance(a, AbstractNode):
        a.values, a.partials = wrap(x=a.values), wrap(x=a.partials)
        return a

    return Node(values=wrap(x=a))


def ravel(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
          order: Optional[str] = 'C') -> AbstractNode:
    def wrap(x):
        return npw.numpy.ravel(a=x, order=order)

    if isinstance(a, AbstractNode):
        a.values, a.partials = wrap(x=a.values), wrap(x=a.partials)
        return a

    return Node(values=wrap(x=a))


def moveaxis(a: Union[npw.ndarray, AbstractNode],
             source: Union[int, Iterable[int]],
             destination: Union[int, Iterable[int]]) -> AbstractNode:
    def wrap(x):
        return npw.numpy.moveaxis(a=x, source=source, destination=destination)

    if isinstance(a, AbstractNode):
        a.values, a.partials = wrap(x=a.values), wrap(x=a.partials)
        return a

    return Node(values=wrap(x=a))


def swapaxes(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
             axis1: int,
             axis2: int) -> AbstractNode:
    def wrap(x):
        return npw.numpy.swapaxes(a=x, axis1=axis1, axis2=axis2)

    if isinstance(a, AbstractNode):
        a.values, a.partials = wrap(x=a.values), wrap(x=a.partials)
        return a

    return Node(values=wrap(x=a))


def transpose(a: Union[npw.ndarray, Iterable, int, float],
              axes: Union[Iterable, tuple, Iterable[int], None] = None) -> AbstractNode:
    def wrap(x):
        return npw.numpy.transpose(a=x, axes=axes)

    if isinstance(a, AbstractNode):
        a.values, a.partials = wrap(x=a.values), wrap(x=a.partials)
        return a

    return Node(values=wrap(x=a))


# TODO: Implement changing number of dimensions


def concatenate(arrays: Any,
                axis: Optional[int] = None,
                out: Optional[npw.ndarray] = None,
                dtype: Union[str, object] = None,
                casting: Optional[str] = None,
                *args: Any,
                **kwargs: Any) -> AbstractNode:
    def wrap(x):
        return npw.numpy.concatenate(arrays=x, axis=axis, out=out, dtype=dtype, casting=casting, *args, **kwargs)

    values, partials = [], []
    for el in arrays:
        if isinstance(el, AbstractNode):
            values.append(el.values)
            partials.append(el.partials)
        else:
            values.append(el)
            partials.append(npw.numpy.zeros_like(el))

    return Node(values=wrap(values), partials=wrap(partials))
