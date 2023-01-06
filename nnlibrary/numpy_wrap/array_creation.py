from typing import Any
from typing import Iterable
from typing import Optional
from typing import Union

import nnlibrary.numpy_wrap as npw

from .node import AbstractNode
from .node import Node
from .node import node_utils


def empty(shape: Union[int, Iterable, tuple[int]],
          dtype: Optional[object] = None,
          order: Optional[str] = 'C',
          *args: Any,
          **kwargs: Any) -> AbstractNode:
    return Node(values=npw.numpy.empty(shape=shape, dtype=dtype, order=order, *args, **kwargs))


def empty_like(prototype: Union[npw.ndarray, Iterable, int, float, AbstractNode],
               dtype: Optional[object] = None,
               order: Optional[str] = None,
               subok: Optional[bool] = None,
               shape: Union[int, Iterable[int], None] = None) -> AbstractNode:
    prototype = node_utils.get_values_if_needed(x=prototype)
    return Node(values=npw.numpy.empty_like(
        prototype=prototype, dtype=dtype, order=order, subok=subok, shape=shape))


# noinspection PyPep8Naming
def eye(N: int,
        M: Optional[int] = None,
        k: Optional[int] = 0,
        dtype: Optional[object] = float,
        order: Optional[str] = 'C',
        *args: Any,
        **kwargs: Any) -> AbstractNode:
    return Node(values=npw.numpy.eye(N=N, M=M, k=k, dtype=dtype, order=order, *args, **kwargs))


def identity(n: int,
             dtype: Optional[object] = None,
             *args: Any,
             **kwargs: Any) -> AbstractNode:
    return Node(values=npw.numpy.identity(n=n, dtype=dtype, *args, **kwargs))


def ones(shape: Union[int, Iterable, tuple[int]],
         dtype: Optional[object] = None,
         order: Optional[str] = 'C',
         *args: Any,
         **kwargs: Any) -> AbstractNode:
    return Node(values=npw.numpy.ones(shape=shape, dtype=dtype, order=order, *args, **kwargs))


def ones_like(prototype: Union[npw.ndarray, Iterable, int, float, AbstractNode],
              dtype: Optional[object] = None,
              order: Optional[str] = None,
              subok: Optional[bool] = None,
              shape: Union[int, Iterable[int], None] = None) -> AbstractNode:
    prototype = node_utils.get_values_if_needed(x=prototype)
    return Node(values=npw.numpy.ones_like(
        prototype=prototype, dtype=dtype, order=order, subok=subok, shape=shape))


def zeros(shape: Union[int, Iterable, tuple[int]],
          dtype: Optional[object] = None,
          order: Optional[str] = 'C',
          *args: Any,
          **kwargs: Any) -> AbstractNode:
    return Node(values=npw.numpy.zeros(shape=shape, dtype=dtype, order=order, *args, **kwargs))


def zeros_like(prototype: Union[npw.ndarray, Iterable, int, float, AbstractNode],
               dtype: Optional[object] = None,
               order: Optional[str] = None,
               subok: Optional[bool] = None,
               shape: Union[int, Iterable[int], None] = None) -> AbstractNode:
    prototype = node_utils.get_values_if_needed(x=prototype)
    return Node(values=npw.numpy.zeros_like(
        prototype=prototype, dtype=dtype, order=order, subok=subok, shape=shape))


def full(shape: Union[int, Iterable, tuple[int]],
         dtype: Optional[object] = None,
         order: Optional[str] = 'C',
         *args: Any,
         **kwargs: Any) -> AbstractNode:
    return Node(values=npw.numpy.full(shape=shape, dtype=dtype, order=order, *args, **kwargs))


def full_like(prototype: Union[npw.ndarray, Iterable, int, float, AbstractNode],
              dtype: Optional[object] = None,
              order: Optional[str] = None,
              subok: Optional[bool] = None,
              shape: Union[int, Iterable[int], None] = None) -> AbstractNode:
    prototype = node_utils.get_values_if_needed(x=prototype)
    return Node(values=npw.numpy.full_like(
        prototype=prototype, dtype=dtype, order=order, subok=subok, shape=shape))


def array(p_object: Union[npw.ndarray, Iterable, int, float, AbstractNode],
          dtype: Optional[object] = None,
          *args: Any,
          **kwargs: Any) -> AbstractNode:
    p_object = node_utils.get_values_if_needed(x=p_object)
    return Node(values=npw.numpy.array(p_object=p_object, dtype=dtype, *args, **kwargs))


def asarray(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
            dtype: Optional[object] = None,
            order: Optional[str] = None,
            *args: Any,
            **kwargs: Any) -> AbstractNode:
    a = node_utils.get_values_if_needed(x=a)
    return Node(values=npw.numpy.asarray(a=a, dtype=dtype, order=order, *args, **kwargs))


def copy(a: Union[npw.ndarray, Iterable, int, float, AbstractNode],
         order: Optional[str] = 'K',
         subok: Optional[bool] = False) -> AbstractNode:
    a = node_utils.get_values_if_needed(x=a)
    return Node(values=npw.numpy.copy(a=a, order=order, subok=subok))


def arange(start: Optional[int] = None, *args: Any, **kwargs: Any) -> AbstractNode:
    return Node(values=npw.numpy.arange(start=start, *args, **kwargs))


def linspace(start: Union[npw.ndarray, Iterable, int, float, AbstractNode],
             stop: Union[npw.ndarray, Iterable, int, float, AbstractNode],
             num: Optional[int] = 50,
             endpoint: Optional[bool] = True,
             retstep: Optional[bool] = False,
             dtype: Optional[object] = None,
             axis: Optional[int] = 0) -> Any:
    start = node_utils.get_values_if_needed(x=start)
    stop = node_utils.get_values_if_needed(x=stop)
    result = npw.numpy.linspace(
        start=start, stop=stop, num=num, endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis)
    return (Node(values=result[0]), result[1]) if retstep else Node(values=result)


def logspace(start: Union[npw.ndarray, Iterable, int, float, AbstractNode],
             stop: Union[npw.ndarray, Iterable, int, float, AbstractNode],
             num: Optional[int] = 50,
             endpoint: Optional[bool] = True,
             base: Union[npw.ndarray, Iterable, int, float, None] = 10.0,
             dtype: object = None,
             axis: Optional[int] = 0) -> AbstractNode:
    start = node_utils.get_values_if_needed(x=start)
    stop = node_utils.get_values_if_needed(x=stop)
    return Node(values=npw.numpy.logspace(
        start=start, stop=stop, num=num, endpoint=endpoint, base=base, dtype=dtype, axis=axis))


def geomspace(start: Union[npw.ndarray, Iterable, int, float, AbstractNode],
              stop: Union[npw.ndarray, Iterable, int, float, AbstractNode],
              num: Optional[int] = 50,
              endpoint: Optional[bool] = True,
              dtype: object = None,
              axis: Optional[int] = 0) -> AbstractNode:
    start = node_utils.get_values_if_needed(x=start)
    stop = node_utils.get_values_if_needed(x=stop)
    return Node(values=npw.numpy.geomspace(
        start=start, stop=stop, num=num, endpoint=endpoint, dtype=dtype, axis=axis))


# noinspection PyShadowingNames
def meshgrid(*xi: Any,
             copy: Optional[bool] = True,
             sparse: Optional[bool] = False,
             indexing: Optional[str] = 'xy') -> AbstractNode:
    xi = tuple([node_utils.get_values_if_needed(x=x) for x in xi])
    return Node(values=npw.numpy.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing))


def diag(v: Union[npw.ndarray, Iterable, int, float, AbstractNode],
         k: Optional[int] = 0) -> AbstractNode:
    v = node_utils.get_values_if_needed(x=v)
    return Node(values=npw.numpy.diag(v=v, k=k))


def diagflat(v: Union[npw.ndarray, Iterable, int, float, AbstractNode],
             k: Optional[int] = 0) -> AbstractNode:
    v = node_utils.get_values_if_needed(x=v)
    return Node(values=npw.numpy.diagflat(v=v, k=k))


# noinspection PyPep8Naming
def tri(N: int,
        M: Optional[int] = None,
        k: Optional[int] = 0,
        dtype: Optional[object] = float,
        *args: Any,
        **kwargs: Any) -> AbstractNode:
    return Node(values=npw.numpy.tri(N=N, M=M, k=k, dtype=dtype, *args, **kwargs))


def tril(m: Union[npw.ndarray, Iterable, int, float, AbstractNode],
         k: Optional[int] = 0) -> AbstractNode:
    m = node_utils.get_values_if_needed(x=m)
    return Node(values=npw.numpy.tril(m=m, k=k))


def triu(m: Union[npw.ndarray, Iterable, int, float, AbstractNode],
         k: Optional[int] = 0) -> AbstractNode:
    m = node_utils.get_values_if_needed(x=m)
    return Node(values=npw.numpy.triu(m=m, k=k))


# noinspection PyPep8Naming
def vander(x: Union[npw.ndarray, Iterable, int, float],
           N: Optional[int] = None,
           increasing: Optional[bool] = False) -> AbstractNode:
    x = node_utils.get_values_if_needed(x=x)
    return Node(values=npw.numpy.vander(x=x, N=N, increasing=increasing))
