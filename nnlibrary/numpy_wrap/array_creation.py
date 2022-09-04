from typing import Any
from typing import Iterable
from typing import Optional
from typing import Union

import nnlibrary.numpy_wrap as npw

from .node import AbstractNode
from .node import Node


def empty(shape: Union[int, Iterable, tuple[int]],
          dtype: Optional[object] = None,
          order: Optional[str] = 'C',
          *args: Any,
          **kwargs: Any) -> AbstractNode:
    return Node(values=npw.numpy.empty(shape=shape, dtype=dtype, order=order, *args, **kwargs))


def empty_like(prototype: Union[npw.ndarray, Iterable, int, float],
               dtype: Optional[object] = None,
               order: Optional[str] = None,
               subok: Optional[bool] = None,
               shape: Union[int, Iterable[int], None] = None) -> AbstractNode:
    return Node(values=npw.numpy.empty_like(prototype=prototype, dtype=dtype, order=order, subok=subok, shape=shape))


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


def ones_like(prototype: Union[npw.ndarray, Iterable, int, float],
              dtype: Optional[object] = None,
              order: Optional[str] = None,
              subok: Optional[bool] = None,
              shape: Union[int, Iterable[int], None] = None) -> AbstractNode:
    return Node(values=npw.numpy.ones_like(prototype=prototype, dtype=dtype, order=order, subok=subok, shape=shape))


def zeros(shape: Union[int, Iterable, tuple[int]],
          dtype: Optional[object] = None,
          order: Optional[str] = 'C',
          *args: Any,
          **kwargs: Any) -> AbstractNode:
    return Node(values=npw.numpy.zeros(shape=shape, dtype=dtype, order=order, *args, **kwargs))


def zeros_like(prototype: Union[npw.ndarray, Iterable, int, float],
               dtype: Optional[object] = None,
               order: Optional[str] = None,
               subok: Optional[bool] = None,
               shape: Union[int, Iterable[int], None] = None) -> AbstractNode:
    return Node(values=npw.numpy.zeros_like(prototype=prototype, dtype=dtype, order=order, subok=subok, shape=shape))


def full(shape: Union[int, Iterable, tuple[int]],
         dtype: Optional[object] = None,
         order: Optional[str] = 'C',
         *args: Any,
         **kwargs: Any) -> AbstractNode:
    return Node(values=npw.numpy.full(shape=shape, dtype=dtype, order=order, *args, **kwargs))


def full_like(prototype: Union[npw.ndarray, Iterable, int, float],
              dtype: Optional[object] = None,
              order: Optional[str] = None,
              subok: Optional[bool] = None,
              shape: Union[int, Iterable[int], None] = None) -> AbstractNode:
    return Node(values=npw.numpy.full_like(prototype=prototype, dtype=dtype, order=order, subok=subok, shape=shape))

