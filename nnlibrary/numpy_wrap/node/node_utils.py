from typing import Any
from typing import Union

import nnlibrary.numpy_wrap as npw

from . import node


def get_values_if_needed(x: Any) -> Any:
    return x.values if isinstance(x, node.AbstractNode) else x


def convert_to_node_if_needed(x: Any) -> node.AbstractNode:
    return x if isinstance(x, node.AbstractNode) else node.Node(values=x)


def unpack_node_if_needed(x: Union[node.AbstractNode, npw.ndarray]) -> node.AbstractNode:
    # if isinstance(x, node.AbstractNode):
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
