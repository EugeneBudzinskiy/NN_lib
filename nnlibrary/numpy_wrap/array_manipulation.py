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
                out: Optional[Union[npw.ndarray]] = None,
                dtype: Union[str, object] = None,
                casting: Optional[str] = None,
                *args: Any,
                **kwargs: Any) -> AbstractNode:
    def wrap(x, o):
        return npw.numpy.concatenate(arrays=x, axis=axis, out=o, dtype=dtype, casting=casting, *args, **kwargs)

    values, partials = [], []
    for el in arrays:
        if isinstance(el, AbstractNode):
            values.append(el.values)
            partials.append(el.partials)
        else:
            values.append(el)
            partials.append(npw.numpy.zeros_like(el))

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = wrap(x=values, o=out.values), wrap(x=partials, o=out.partials)
        return out

    return Node(values=wrap(x=values, o=out), partials=wrap(x=partials, o=out))


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


def stack(arrays: Union[Iterable[npw.ndarray], Iterable, int, float, Iterable[AbstractNode]],
          axis: Optional[int] = 0,
          out: Optional[Union[npw.ndarray, AbstractNode]] = None) -> AbstractNode:
    def wrap(x, o):
        return npw.numpy.stack(arrays=x, axis=axis, out=o)

    values, partials = [], []
    for el in arrays:
        if isinstance(el, AbstractNode):
            values.append(el.values)
            partials.append(el.partials)
        else:
            values.append(el)
            partials.append(npw.numpy.zeros_like(el))

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = wrap(x=values, o=out.values), wrap(x=partials, o=out.partials)
        return out

    return Node(values=wrap(x=values, o=out), partials=wrap(x=partials, o=out))


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


def delete(arr: Union[npw.ndarray, Iterable, int, float, AbstractNode],
           obj: Union[slice, int, npw.ndarray, Iterable, float, AbstractNode],
           axis: Optional[int] = None) -> AbstractNode:
    obj = node_utils.get_values_if_needed(x=obj)

    def wrap(x):
        return npw.numpy.delete(arr=x, obj=obj, axis=axis)

    if isinstance(arr, AbstractNode):
        return Node(values=wrap(x=arr.values), partials=wrap(x=arr.partials))

    return Node(values=wrap(x=arr))


def insert(arr: Union[npw.ndarray, Iterable, int, float, AbstractNode],
           obj: Union[int, slice, Iterable[int], AbstractNode],
           values: Union[npw.ndarray, Iterable, int, float, AbstractNode],
           axis: Optional[int] = None) -> AbstractNode:
    obj = node_utils.get_values_if_needed(x=obj)

    def wrap(x, v):
        return npw.numpy.insert(arr=x, obj=obj, values=v, axis=axis)

    arr = node_utils.convert_to_node_if_needed(x=arr)
    values = node_utils.convert_to_node_if_needed(x=values)

    return Node(values=wrap(x=arr.values, v=values.values),
                partials=wrap(x=arr.partials, v=values.partials))


def append(arr: Union[npw.ndarray, Iterable, int, float, AbstractNode],
           values: Union[npw.ndarray, Iterable, int, float, AbstractNode],
           axis: Optional[int] = None) -> AbstractNode:

    def wrap(x, v):
        return npw.numpy.append(arr=x, values=v, axis=axis)

    arr = node_utils.convert_to_node_if_needed(x=arr)
    values = node_utils.convert_to_node_if_needed(x=values)

    return Node(values=wrap(x=arr.values, v=values.values),
                partials=wrap(x=arr.partials, v=values.partials))


def resize(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
           new_shape: Union[int, Iterable, tuple[int]]) -> AbstractNode:

    def wrap(x):
        return npw.numpy.resize(a=x, new_shape=new_shape)

    if isinstance(a, AbstractNode):
        return Node(values=wrap(x=a.values), partials=wrap(x=a.partials))

    return Node(values=wrap(x=a))


def flip(m: Union[npw.ndarray, Iterable, int, float, AbstractNode],
         axis: Union[None, int, Iterable, tuple[int]] = None) -> AbstractNode:

    def wrap(x):
        return npw.numpy.flip(m=x, axis=axis)

    if isinstance(m, AbstractNode):
        m.values, m.partials = wrap(x=m.values), wrap(x=m.partials)
        return m

    return Node(values=wrap(x=m))


def fliplr(m: Union[npw.ndarray, Iterable, int, float, AbstractNode]) -> AbstractNode:
    def wrap(x):
        return npw.numpy.fliplr(m=x)

    if isinstance(m, AbstractNode):
        m.values, m.partials = wrap(x=m.values), wrap(x=m.partials)
        return m

    return Node(values=wrap(x=m))


def flipud(m: Union[npw.ndarray, Iterable, int, float, AbstractNode]) -> AbstractNode:
    def wrap(x):
        return npw.numpy.flipud(m=x)

    if isinstance(m, AbstractNode):
        m.values, m.partials = wrap(x=m.values), wrap(x=m.partials)
        return m

    return Node(values=wrap(x=m))


def roll(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
         shift: Union[int, Iterable, tuple[int]],
         axis: Union[int, Iterable, tuple[int], None] = None) -> AbstractNode:
    def wrap(x):
        return npw.numpy.roll(a=x, shift=shift, axis=axis)

    if isinstance(a, AbstractNode):
        a.values, a.partials = wrap(x=a.values), wrap(x=a.partials)
        return a

    return Node(values=wrap(x=a))


def rot90(m: Union[npw.ndarray, Iterable, int, float, AbstractNode],
          k: int = 1,
          axes: Any = (0, 1)) -> AbstractNode:
    def wrap(x):
        return npw.numpy.rot90(m=x, k=k, axes=axes)

    if isinstance(m, AbstractNode):
        m.values, m.partials = wrap(x=m.values), wrap(x=m.partials)
        return m

    return Node(values=wrap(x=m))


