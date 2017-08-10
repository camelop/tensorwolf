# profile""" define the behaviors of excutor """
from __future__ import absolute_import
import numpy as np
from tensorwolf.topo import *
from tensorwolf.ops import *
import sys
import os
#reference: dlsys-autodiff


class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""

    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        topo_order: list of nodes in topological order
        """
        self.eval_node_list = eval_node_list
        self.topo_order = find_topo_sort(self.eval_node_list)

    # profile
    def run(self, feed_dict):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list. NDArray or np.ndarray.
        """
        node_to_val_map = {}
        for node, value in feed_dict.items():
            node_to_val_map[node] = np.array(value)

        for node in self.topo_order:
            if node in node_to_val_map:
                continue
            #print("Compute: ", node.name)
            #print("Compute-Type: ", type(node.op))
            input_vals = [node_to_val_map[n] for n in node.inputs]
            value = node.op.compute(node, input_vals)
            # if isinstance(value, np.ndarray):
            #    print("shape:", value.shape)
            node_to_val_map[node] = value
            # os.system("PAUSE")

        return [node_to_val_map[n] for n in self.eval_node_list]


def gradients(output_node, node_list):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        input_grads_list = node.op.gradient(node, output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])
    '''
    print("node_list: ")
    for i in node_list:
        print("  ", i)
    print("node_to_output_grad")
    for i in node_to_output_grad:
        print("  ", i)
    '''
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list


def sum_node_list(node_list):
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
