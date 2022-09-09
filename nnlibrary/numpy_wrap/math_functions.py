from typing import Any
from typing import Union
from typing import Optional
from typing import Iterable

import nnlibrary.numpy_wrap as npw

from nnlibrary.numpy_wrap.node import AbstractNode
from nnlibrary.numpy_wrap.node import Node
from nnlibrary.numpy_wrap.node import node_utils


# TODO: Implement support for `args` and `kwargs` for most functions bellow (ignored for now)


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


# noinspection PyProtectedMember
def prod(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
         axis: Union[None, int, Iterable, tuple[int]] = None,
         dtype: Optional[object] = None,
         out: Optional[Union[npw.ndarray, AbstractNode]] = None,
         keepdims: Optional[bool] = npw.numpy._NoValue,
         initial: Union[int, float, complex, None] = npw.numpy._NoValue,
         where: Union[npw.ndarray, Iterable, int, float, None, AbstractNode] = npw.numpy._NoValue) -> AbstractNode:
    where = node_utils.get_values_if_needed(x=where)

    def wrap(x, o):
        return npw.numpy.prod(a=x, axis=axis, dtype=dtype, out=o, keepdims=keepdims, initial=initial, where=where)

    a = node_utils.convert_to_node_if_needed(x=a)

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = wrap(x=a.values, o=out.values), wrap(x=a.partials, o=out.partials)
        return out

    return Node(values=wrap(x=a.values, o=out), partials=wrap(x=a.partials, o=out))


# noinspection PyProtectedMember, PyShadowingBuiltins
def sum(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
        axis: Union[None, int, Iterable, tuple[int]] = None,
        dtype: Optional[object] = None,
        out: Optional[Union[npw.ndarray, AbstractNode]] = None,
        keepdims: Optional[bool] = npw.numpy._NoValue,
        initial: Union[int, float, complex, None] = npw.numpy._NoValue,
        where: Union[npw.ndarray, Iterable, int, float, None, AbstractNode] = npw.numpy._NoValue) -> AbstractNode:
    where = node_utils.get_values_if_needed(x=where)

    def wrap(x, o):
        return npw.numpy.sum(a=x, axis=axis, dtype=dtype, out=o, keepdims=keepdims, initial=initial, where=where)

    a = node_utils.convert_to_node_if_needed(x=a)

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = wrap(x=a.values, o=out.values), wrap(x=a.partials, o=out.partials)
        return out

    return Node(values=wrap(x=a.values, o=out), partials=wrap(x=a.partials, o=out))


# noinspection PyProtectedMember
def nanprod(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
            axis: Union[None, int, Iterable, tuple[int]] = None,
            dtype: Optional[object] = None,
            out: Optional[Union[npw.ndarray, AbstractNode]] = None,
            keepdims: Optional[bool] = npw.numpy._NoValue,
            initial: Union[int, float, complex, None] = npw.numpy._NoValue,
            where: Union[npw.ndarray, Iterable, int, float, None, AbstractNode] = npw.numpy._NoValue) -> AbstractNode:
    where = node_utils.get_values_if_needed(x=where)

    def wrap(x, o):
        return npw.numpy.nanprod(a=x, axis=axis, dtype=dtype, out=o, keepdims=keepdims, initial=initial, where=where)

    a = node_utils.convert_to_node_if_needed(x=a)

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = wrap(x=a.values, o=out.values), wrap(x=a.partials, o=out.partials)
        return out

    return Node(values=wrap(x=a.values, o=out), partials=wrap(x=a.partials, o=out))


# noinspection PyProtectedMember
def nansum(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
           axis: Union[None, int, Iterable, tuple[int]] = None,
           dtype: Optional[object] = None,
           out: Optional[Union[npw.ndarray, AbstractNode]] = None,
           keepdims: Optional[bool] = npw.numpy._NoValue,
           initial: Union[int, float, complex, None] = npw.numpy._NoValue,
           where: Union[npw.ndarray, Iterable, int, float, None, AbstractNode] = npw.numpy._NoValue) -> AbstractNode:
    where = node_utils.get_values_if_needed(x=where)

    def wrap(x, o):
        return npw.numpy.nansum(a=x, axis=axis, dtype=dtype, out=o, keepdims=keepdims, initial=initial, where=where)

    a = node_utils.convert_to_node_if_needed(x=a)

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = wrap(x=a.values, o=out.values), wrap(x=a.partials, o=out.partials)
        return out

    return Node(values=wrap(x=a.values, o=out), partials=wrap(x=a.partials, o=out))


def cumprod(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
            axis: Optional[int] = None,
            dtype: Optional[object] = None,
            out: Optional[Union[npw.ndarray, AbstractNode]] = None) -> AbstractNode:
    def wrap(x, o):
        return npw.numpy.cumprod(a=x, axis=axis, dtype=dtype, out=o)

    a = node_utils.convert_to_node_if_needed(x=a)

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = wrap(x=a.values, o=out.values), wrap(x=a.partials, o=out.partials)
        return out

    return Node(values=wrap(x=a.values, o=out), partials=wrap(x=a.partials, o=out))


def cumsum(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
           axis: Optional[int] = None,
           dtype: Optional[object] = None,
           out: Optional[Union[npw.ndarray, AbstractNode]] = None) -> AbstractNode:
    def wrap(x, o):
        return npw.numpy.cumsum(a=x, axis=axis, dtype=dtype, out=o)

    a = node_utils.convert_to_node_if_needed(x=a)

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = wrap(x=a.values, o=out.values), wrap(x=a.partials, o=out.partials)
        return out

    return Node(values=wrap(x=a.values, o=out), partials=wrap(x=a.partials, o=out))


def nancumprod(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
               axis: Optional[int] = None,
               dtype: Optional[object] = None,
               out: Optional[Union[npw.ndarray, AbstractNode]] = None) -> AbstractNode:
    def wrap(x, o):
        return npw.numpy.nancumprod(a=x, axis=axis, dtype=dtype, out=o)

    a = node_utils.convert_to_node_if_needed(x=a)

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = wrap(x=a.values, o=out.values), wrap(x=a.partials, o=out.partials)
        return out

    return Node(values=wrap(x=a.values, o=out), partials=wrap(x=a.partials, o=out))


def nancumsum(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
              axis: Optional[int] = None,
              dtype: Optional[object] = None,
              out: Optional[Union[npw.ndarray, AbstractNode]] = None) -> AbstractNode:
    def wrap(x, o):
        return npw.numpy.nancumsum(a=x, axis=axis, dtype=dtype, out=o)

    a = node_utils.convert_to_node_if_needed(x=a)

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = wrap(x=a.values, o=out.values), wrap(x=a.partials, o=out.partials)
        return out

    return Node(values=wrap(x=a.values, o=out), partials=wrap(x=a.partials, o=out))


def cross(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
          b: Union[npw.ndarray, Iterable, int, float, AbstractNode],
          axisa: Optional[int] = -1,
          axisb: Optional[int] = -1,
          axisc: Optional[int] = -1,
          axis: Optional[int] = None) -> AbstractNode:
    a = node_utils.convert_to_node_if_needed(x=a)
    b = node_utils.convert_to_node_if_needed(x=b)

    values = npw.numpy.cross(a=a.values, b=b.values, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)
    partials = npw.numpy.cross(a=a.partials, b=b.values, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis) + \
               npw.numpy.cross(a=a.values, b=b.partials, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)

    return Node(values=values, partials=partials)


def exp(x: Union[npw.ndarray, Iterable, AbstractNode],
        *args: Any,
        **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.exp(x=x.values, *args, **kwargs)
    partials = x.partials * values
    return Node(values=values, partials=partials)


def expm1(x: Union[npw.ndarray, Iterable, AbstractNode],
          *args: Any,
          **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.expm1(x=x.values, *args, **kwargs)
    partials = x.partials * npw.numpy.exp(x=x.values, *args, **kwargs)
    return Node(values=values, partials=partials)


def exp2(x: Union[npw.ndarray, Iterable, AbstractNode],
         *args: Any,
         **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.exp2(x=x.values, *args, **kwargs)
    partials = x.partials * values * npw.numpy.log(2)
    return Node(values=values, partials=partials)


def log(x: Union[npw.ndarray, Iterable, AbstractNode],
        *args: Any,
        **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.log(x=x.values, *args, **kwargs)
    partials = x.partials / x.values
    return Node(values=values, partials=partials)


def log2(x: Union[npw.ndarray, Iterable, AbstractNode],
         *args: Any,
         **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.log2(x=x.values, *args, **kwargs)
    partials = x.partials / (x.values * npw.numpy.log(2))
    return Node(values=values, partials=partials)


def log10(x: Union[npw.ndarray, Iterable, AbstractNode],
          *args: Any,
          **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.log10(x=x.values, *args, **kwargs)
    partials = x.partials / (x.values * npw.numpy.log(10))
    return Node(values=values, partials=partials)


def log1p(x: Union[npw.ndarray, Iterable, AbstractNode],
          *args: Any,
          **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.log1p(x=x.values, *args, **kwargs)
    partials = x.partials / (1 + x.values)
    return Node(values=values, partials=partials)


def logaddexp(x1: Union[npw.ndarray, Iterable, AbstractNode],
              x2: Union[npw.ndarray, Iterable, AbstractNode],
              *args: Any,
              **kwargs: Any) -> AbstractNode:
    x1 = node_utils.convert_to_node_if_needed(x=x1)
    x2 = node_utils.convert_to_node_if_needed(x=x2)

    values = npw.numpy.logaddexp(x1=x1.values, x2=x2.values, *args, **kwargs)
    e_x1, e_x2 = npw.numpy.exp(x=x1), npw.numpy.exp(x=x2)
    partials = (x1.partials * e_x1 + x2.partials * e_x2) / (e_x1 + e_x2)
    return Node(values=values, partials=partials)


def logaddexp2(x1: Union[npw.ndarray, Iterable, AbstractNode],
               x2: Union[npw.ndarray, Iterable, AbstractNode],
               *args: Any,
               **kwargs: Any) -> AbstractNode:
    x1 = node_utils.convert_to_node_if_needed(x=x1)
    x2 = node_utils.convert_to_node_if_needed(x=x2)

    values = npw.numpy.logaddexp2(x1=x1.values, x2=x2.values, *args, **kwargs)
    e2_x1, e2_x2 = npw.numpy.exp2(x=x1), npw.numpy.exp2(x=x2)
    partials = (x1.partials * e2_x1 + x2.partials * e2_x2) / (e2_x1 + e2_x2)
    return Node(values=values, partials=partials)


def add(x1: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
        x2: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
        *args: Any,
        **kwargs: Any) -> AbstractNode:
    x1 = node_utils.convert_to_node_if_needed(x=x1)
    x2 = node_utils.convert_to_node_if_needed(x=x2)

    values = npw.numpy.add(x1=x1.values, x2=x2.values, *args, **kwargs)
    partials = x1.partials + x2.partials
    return Node(values=values, partials=partials)


def positive(x: Union[npw.ndarray, Iterable, int, float, complex, AbstractNode],
             *args: Any,
             **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.positive(x=x.values, *args, **kwargs)
    partials = + x.partials
    return Node(values=values, partials=partials)


def negative(x: Union[npw.ndarray, Iterable, int, float, complex, AbstractNode],
             *args: Any,
             **kwargs: Any) -> AbstractNode:
    x = node_utils.convert_to_node_if_needed(x=x)
    values = npw.numpy.negative(x=x.values, *args, **kwargs)
    partials = - x.partials
    return Node(values=values, partials=partials)


def multiply(x1: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
             x2: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
             *args: Any,
             **kwargs: Any) -> AbstractNode:
    x1 = node_utils.convert_to_node_if_needed(x=x1)
    x2 = node_utils.convert_to_node_if_needed(x=x2)

    values = npw.numpy.multiply(x1=x1.values, x2=x2.values, *args, **kwargs)
    partials = x1.partials * x2.values + x1.values * x2.partials
    return Node(values=values, partials=partials)


def divide(x1: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
           x2: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
           *args: Any,
           **kwargs: Any) -> AbstractNode:
    x1 = node_utils.convert_to_node_if_needed(x=x1)
    x2 = node_utils.convert_to_node_if_needed(x=x2)

    values = npw.numpy.divide(x1=x1.values, x2=x2.values, *args, **kwargs)
    partials = (x1.partials * x2.values - x1.values * x2.partials) / x2.values ** 2
    return Node(values=values, partials=partials)


def power(x1: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
          x2: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
          *args: Any,
          **kwargs: Any) -> AbstractNode:
    x1 = node_utils.convert_to_node_if_needed(x=x1)
    x2 = node_utils.convert_to_node_if_needed(x=x2)

    values = npw.numpy.power(x1=x1.values, x2=x2.values, *args, **kwargs)
    par_x1 = x1.partials * x2.values * npw.numpy.power(x1.values, x2.values - 1)
    par_x2 = x2.partials * values * npw.numpy.log(npw.numpy.abs(x1.values))
    partials = par_x1 + par_x2  # TODO: Add offset when x near 0 (replace it with some epsilon)
    return Node(values=values, partials=partials)


def subtract(x1: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
             x2: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
             *args: Any,
             **kwargs: Any) -> AbstractNode:
    x1 = node_utils.convert_to_node_if_needed(x=x1)
    x2 = node_utils.convert_to_node_if_needed(x=x2)

    values = npw.numpy.subtract(x1=x1.values, x2=x2.values, *args, **kwargs)
    partials = x1.partials - x2.partials
    return Node(values=values, partials=partials)


def true_divide(x1: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
                x2: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
                *args: Any,
                **kwargs: Any) -> AbstractNode:
    return npw.divide(x1=x1, x2=x2, *args, **kwargs)  # TODO: Rewrite it as actual implementation


def maximum(x1: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
            x2: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
            *args: Any,
            **kwargs: Any) -> AbstractNode:
    x1 = node_utils.convert_to_node_if_needed(x=x1)
    x2 = node_utils.convert_to_node_if_needed(x=x2)

    values = npw.numpy.maximum(x1=x1.values, x2=x2.values, *args, **kwargs)
    partials = npw.numpy.where(condition=x1.values >= x2.values, x=x1.partials, y=x2.partials)
    return Node(values=values, partials=partials)


def fmax(x1: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
         x2: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
         *args: Any,
         **kwargs: Any) -> AbstractNode:
    x1 = node_utils.convert_to_node_if_needed(x=x1)
    x2 = node_utils.convert_to_node_if_needed(x=x2)

    values = npw.numpy.fmax(x1=x1.values, x2=x2.values, *args, **kwargs)
    partials = npw.numpy.where(condition=x1.values >= x2.values, x=x1.partials, y=x2.partials)
    return Node(values=values, partials=partials)


def minimum(x1: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
            x2: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
            *args: Any,
            **kwargs: Any) -> AbstractNode:
    x1 = node_utils.convert_to_node_if_needed(x=x1)
    x2 = node_utils.convert_to_node_if_needed(x=x2)

    values = npw.numpy.minimum(x1=x1.values, x2=x2.values, *args, **kwargs)
    partials = npw.numpy.where(condition=x1.values <= x2.values, x=x1.partials, y=x2.partials)
    return Node(values=values, partials=partials)


def fmin(x1: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
         x2: Union[npw.numpy.Number, npw.ndarray, Iterable, AbstractNode],
         *args: Any,
         **kwargs: Any) -> AbstractNode:
    x1 = node_utils.convert_to_node_if_needed(x=x1)
    x2 = node_utils.convert_to_node_if_needed(x=x2)

    values = npw.numpy.fmin(x1=x1.values, x2=x2.values, *args, **kwargs)
    partials = npw.numpy.where(condition=x1.values <= x2.values, x=x1.partials, y=x2.partials)
    return Node(values=values, partials=partials)


def clip(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
         a_min: Union[npw.ndarray, Iterable, int, float, None, AbstractNode],
         a_max: Union[npw.ndarray, Iterable, int, float, None, AbstractNode],
         out: Optional[Union[npw.ndarray, AbstractNode]] = None,
         **kwargs: Any) -> AbstractNode:
    a = node_utils.convert_to_node_if_needed(x=a)
    partials = a.partials
    a_min_, a_max_ = a_min, a_max

    def wrap(x, o):
        return npw.numpy.clip(a=x, a_max=a_max_, a_min=a_min_, out=o, **kwargs)

    if a_min:
        a_min = node_utils.convert_to_node_if_needed(x=a_min)
        partials = npw.numpy.where(condition=a.values >= a_min, x=partials, y=a_min.partials)
        a_min_ = a_min.values

    if a_max:
        a_max = node_utils.convert_to_node_if_needed(x=a_max)
        partials = npw.numpy.where(condition=a.values <= a_max, x=partials, y=a_max.partials)
        a_max_ = a_max.values

    if out:
        out = node_utils.convert_to_node_if_needed(x=out)
        out.values, out.partials = wrap(x=a.values, o=out.values), partials
        return out

    values = wrap(x=a.values, o=out)
    return Node(values=values, partials=partials)
