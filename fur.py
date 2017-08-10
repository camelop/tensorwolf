""" Some API to make tensorwolf look like tensorflow """
from tensorwolf.executor import *


zeros = np.zeros
ones = np.ones
float32 = np.float32
float64 = np.float64


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None, name=None):
    return_val = np.random.normal(
        loc=mean, scale=stddev, size=shape).astype(dtype)
    # print(return_val)
    return return_val


class Session(object):
    def __call__(self, name="Session"):
        """ Just a shell, nothing else."""
        newSession = Session()
        newSession.name = name
        newSession.ex = None
        return newSession

    def run(self, eval_node_list, feed_dict={}):
        isList = True
        if not isinstance(eval_node_list, list):
            isList = False
            eval_node_list = [eval_node_list]
        self.ex = Executor(eval_node_list=eval_node_list)
        if isList:
            return self.ex.run(feed_dict=feed_dict)
        else:
            return self.ex.run(feed_dict=feed_dict)[0]

    def __enter__(self):
        return self

    def __exit__(self, e_t, e_v, t_b):
        # I do not know what these args mean...
        return


import tensorwolf.topo as topo


class train(object):
    class Optimizer(object):
        def __init__(self):
            return None

        def get_variables_list(self):
            variables_list = []
            for variable in variable_to_node:
                variables_list.append(variable)
            return variables_list

    class GradientDescentOptimizer(Optimizer):
        def __init__(self, learning_rate=0.01, name="GradientDescentOptimizer"):
            self.learning_rate = learning_rate
            self.name = name

        def minimize(self, target):
            variables_prepare = self.get_variables_list()
            variables_to_change = []
            used_ones = topo.find_topo_sort(node_list=[target])
            for v in variables_prepare:
                if v in used_ones:
                    variables_to_change.append(v)
            variables_gradients = gradients(target, variables_to_change)
            change_list = []
            for index, variable in enumerate(variables_to_change):
                change_list.append(
                    assign(variable, variable - (self.learning_rate * variables_gradients[index])))
            return pack(change_list)

    class AdamOptimizer(Optimizer):
        def __init__(self, learning_rate=0.001,
                     beta1=0.9,
                     beta2=0.999,
                     epsilon=1e-8,
                     name="AdamOptimizer"):
            # for more detail:
            # https://arxiv.org/abs/1412.6980
            # https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.epsilon = epsilon
            self.name = name

        def minimize(self, target):
            variables_to_change = self.get_variables_list()
            variables_gradients = gradients(target, variables_to_change)
            change_list = []
            # use constant to avoid initialize
            self.t = constant(0)
            self.assignt = assign(self.t, self.t + 1)
            self.lrt = self.learning_rate * \
                sqrt_op(1 - pow_op(constant(self.beta2), self.assignt)) / \
                (1 - pow_op(constant(self.beta1), self.assignt))
            # initialize m & v for globel variables
            # also use constant to avoid initialize
            self.m = []
            self.assignm = []
            self.v = []
            self.assignv = []
            for variable in variables_to_change:
                self.m.append(constant(0))
                self.v.append(constant(0))
            # update global variables
            for index, variable in enumerate(variables_to_change):
                # construct the new value
                g = variables_gradients[index]
                nw_m = self.m[index]
                mt = assign(nw_m, nw_m * self.beta1 + g * (1 - self.beta1))
                nw_v = self.v[index]
                vt = assign(nw_v, nw_v * self.beta2 + g * g * (1 - self.beta2))
                newValue = variable - self.lrt * mt / \
                    (sqrt_op(vt) + constant(self.epsilon))
                # add the assign operator into change list
                change_list.append(assign(variable, newValue))
            return pack(change_list)


class nn(object):
    """ Supports neural network. """
    class SoftmaxOp(Op):
        def __call__(self, node_A, dim=-1, name=None):
            if name is None:
                name = "Softmax(%s, dim=%s)" % (node_A.name, dim)
            exp_node_A = exp(node_A)
            new_node = exp_node_A / \
                broadcastto_op(reduce_sum(exp_node_A, axis=dim), exp_node_A)
            new_node.name = name
            return new_node

    softmax = SoftmaxOp()
    relu = relu

    class SoftmaxCrossEntropyWithLogitsOp(Op):
        def __call__(self, logits, labels):
            return softmax_cross_entropy_op(logits, labels)
            # to be honest the thing above is somehow bad
            # here's an equal expression
            # return (-reduce_sum(labels * log(nn.softmax(logits)), reduction_indices=[1]))

    softmax_cross_entropy_with_logits = SoftmaxCrossEntropyWithLogitsOp()
    conv2d = conv2d_op
    max_pool = max_pool

    class DropoutOp(Op):
        def __call__(self, node_A, node_B, name=None):
            new_node = mul_op(node_A, probshape_op(node_A, node_B)) / node_B
            if name is None:
                name = "Dropout(%s,prob=%s)" % (node_A.name, node_B.name)
            new_node.name = name
            return new_node

    dropout = DropoutOp()
