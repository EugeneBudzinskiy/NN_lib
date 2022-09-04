from typing import Any
from typing import Union
from typing import Callable

import nnlibrary.numpy_wrap as npw

from .node import AbstractNode
from .node import Node


def get_values_if_needed(x: Union[AbstractNode, Any]) -> Union[npw.ndarray, Any]:
    return x.values if isinstance(x, AbstractNode) else x


def convert_to_node_if_needed(x: Union[AbstractNode, Any]) -> AbstractNode:
    return x if isinstance(x, AbstractNode) else Node(values=x)


def unpack_node_if_needed(x: Union[AbstractNode, npw.ndarray[AbstractNode]]) -> AbstractNode:
    # if isinstance(x, AbstractNode):
    #     return x
    #
    # flat = x.flatten()
    # result = npw.zeros_like(flat)
    #
    # for i in range(len(flat)):
    #     result.values[i] = flat[i].values
    #     result.partials[i] = flat[i].partials
    #
    # return result.reshape(x.shape)

    raise NotImplementedError

