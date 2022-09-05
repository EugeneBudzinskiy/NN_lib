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


def block(arrays: Union[list[AbstractNode], Any]) -> AbstractNode:
    def wrap(x):
        return npw.numpy.block(arrays=x)

    values, partials = [], []
    for el in arrays:
        if isinstance(el, AbstractNode):
            values.append(el.values)
            partials.append(el.partials)
        else:
            values.append(el)
            partials.append(npw.numpy.zeros_like(el))

    return Node(values=wrap(values), partials=wrap(partials))


def stack(arrays: Union[Iterable[npw.ndarray], Iterable, int, float, AbstractNode],
          axis: Optional[int] = 0,
          out: Optional[npw.ndarray] = None) -> AbstractNode:
    def wrap(x):
        return npw.numpy.stack(arrays=x, axis=axis, out=out)

    values, partials = [], []
    for el in arrays:
        if isinstance(el, AbstractNode):
            values.append(el.values)
            partials.append(el.partials)
        else:
            values.append(el)
            partials.append(npw.numpy.zeros_like(el))

    return Node(values=wrap(values), partials=wrap(partials))


def vstack(tup: Iterable) -> AbstractNode:
    def wrap(x):
        return npw.numpy.vstack(tup=x)

    values, partials = [], []
    for el in tup:
        if isinstance(el, AbstractNode):
            values.append(el.values)
            partials.append(el.partials)
        else:
            values.append(el)
            partials.append(npw.numpy.zeros_like(el))

    return Node(values=wrap(values), partials=wrap(partials))


def hstack(tup: Iterable) -> AbstractNode:
    def wrap(x):
        return npw.numpy.hstack(tup=x)

    values, partials = [], []
    for el in tup:
        if isinstance(el, AbstractNode):
            values.append(el.values)
            partials.append(el.partials)
        else:
            values.append(el)
            partials.append(npw.numpy.zeros_like(el))

    return Node(values=wrap(values), partials=wrap(partials))


def dstack(tup: Iterable) -> AbstractNode:
    def wrap(x):
        return npw.numpy.dstack(tup=x)

    values, partials = [], []
    for el in tup:
        if isinstance(el, AbstractNode):
            values.append(el.values)
            partials.append(el.partials)
        else:
            values.append(el)
            partials.append(npw.numpy.zeros_like(el))

    return Node(values=wrap(values), partials=wrap(partials))


def column_stack(tup: Iterable) -> AbstractNode:
    def wrap(x):
        return npw.numpy.column_stack(tup=x)

    values, partials = [], []
    for el in tup:
        if isinstance(el, AbstractNode):
            values.append(el.values)
            partials.append(el.partials)
        else:
            values.append(el)
            partials.append(npw.numpy.zeros_like(el))

    return Node(values=wrap(values), partials=wrap(partials))


def row_stack(tup: Iterable) -> AbstractNode:
    def wrap(x):
        return npw.numpy.row_stack(tup=x)

    values, partials = [], []
    for el in tup:
        if isinstance(el, AbstractNode):
            values.append(el.values)
            partials.append(el.partials)
        else:
            values.append(el)
            partials.append(npw.numpy.zeros_like(el))

    return Node(values=wrap(values), partials=wrap(partials))


def split(ary: Union[npw.ndarray, AbstractNode],
          indices_or_sections: Any,
          axis: Optional[int] = 0) -> list[AbstractNode]:
    def wrap(x):
        return npw.numpy.split(ary=x, indices_or_sections=indices_or_sections, axis=axis)

    if isinstance(ary, AbstractNode):
        return [Node(values=v, partials=p) for v, p in zip(wrap(x=ary.values), wrap(x=ary.partials))]

    return [Node(values=v) for v in wrap(x=ary)]


def dsplit(ary: Union[AbstractNode, Any],
           indices_or_sections: Any) -> list[AbstractNode]:
    def wrap(x):
        return npw.numpy.dsplit(ary=x, indices_or_sections=indices_or_sections)

    if isinstance(ary, AbstractNode):
        return [Node(values=v, partials=p) for v, p in zip(wrap(x=ary.values), wrap(x=ary.partials))]

    return [Node(values=v) for v in wrap(x=ary)]


def hsplit(ary: Union[AbstractNode, Any],
           indices_or_sections: Any) -> list[AbstractNode]:
    def wrap(x):
        return npw.numpy.hsplit(ary=x, indices_or_sections=indices_or_sections)

    if isinstance(ary, AbstractNode):
        return [Node(values=v, partials=p) for v, p in zip(wrap(x=ary.values), wrap(x=ary.partials))]

    return [Node(values=v) for v in wrap(x=ary)]


def vsplit(ary: Union[AbstractNode, Any],
           indices_or_sections: Any) -> list[AbstractNode]:
    def wrap(x):
        return npw.numpy.vsplit(ary=x, indices_or_sections=indices_or_sections)

    if isinstance(ary, AbstractNode):
        return [Node(values=v, partials=p) for v, p in zip(wrap(x=ary.values), wrap(x=ary.partials))]

    return [Node(values=v) for v in wrap(x=ary)]


# noinspection PyPep8Naming
def tile(A: Union[npw.ndarray, Iterable, int, float, AbstractNode],
         reps: Union[npw.ndarray, Iterable, int, float, AbstractNode]) -> AbstractNode:
    reps = node_utils.get_values_if_needed(x=reps)

    def wrap(x):
        return npw.numpy.tile(A=x, reps=reps)

    if isinstance(A, AbstractNode):
        return Node(values=wrap(x=A.values), partials=wrap(x=A.partials))

    return Node(values=wrap(x=A))


def repeat(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
           repeats: Union[int, npw.ndarray, Iterable, float, AbstractNode],
           axis: Optional[int] = None) -> AbstractNode:
    repeats = node_utils.get_values_if_needed(x=repeats)

    def wrap(x):
        return npw.numpy.repeat(a=x, repeats=repeats, axis=axis)

    if isinstance(a, AbstractNode):
        return Node(values=wrap(x=a.values), partials=wrap(x=a.partials))

    return Node(values=wrap(x=a))
