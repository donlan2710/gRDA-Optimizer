from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.training import optimizer



class GRDA(optimizer.Optimizer):
    """Optimizer that implements the GRDA algorithm.
    See (https://.......)

    """

    def __init__(self, learning_rate=0.005, c = 0.005, mu=0.7, global_step=0, use_locking=False, name="GRDA"):
        """Construct a new GRDA optimizer.
        Args:
            learning_rate: A Tensor or a floating point value. The 
                learning rate.
            c: A float value or a constant float tensor. Turn on/off the l1 penalty and initial penalty.
            mu: A float value or a constant float tensor. Time expansion of l1 penalty. 
            name: Optional name for the operations created when applying gradients.
            Defaults to "GRDA".
        """
        super(GRDA, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._c = c
        self._mu = mu
        self._global_step = global_step
        self._global_step_on_worker = None
        self._learning_rate_tensor = None


    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                v_ini = random_ops.random_uniform(
                    shape=v.get_shape(), minval = -0.1, maxval = 0.1, dtype=v.dtype.base_dtype, seed = 123)
            self._get_or_make_slot(v, v_ini, "accumulator", self._name)

    def _prepare(self):
        self._learning_rate_tensor = ops.convert_to_tensor(
            self._learning_rate, name="learning_rate")
        with ops.colocate_with(self._learning_rate_tensor):
            self._global_step_on_worker = array_ops.identity(self._global_step) + 1


    def _apply_dense(self, grad, var):
        lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)

        v = self.get_slot(var, "accumulator")
        v_t = state_ops.assign(v, v - lr * grad, use_locking=self._use_locking)

        with ops.device(var.device):
            global_step = math_ops.cast(self._global_step_on_worker, var.dtype.base_dtype)
        c = math_ops.cast(self._c, var.dtype.base_dtype)
        mu = math_ops.cast(self._mu, var.dtype.base_dtype)
        l1 = math_ops.cast(c * math_ops.pow(lr, (0.5 + mu)) * math_ops.pow(global_step, mu), var.dtype.base_dtype)

        # GRDA

        var_update = state_ops.assign(var, math_ops.sign(v_t) * math_ops.maximum(math_ops.abs(v_t) - l1, 0), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, v_t])

    def _apply_sparse(self, grad, var):
        return
        raise NotImplementedError("Sparse gradient updates are not supported yet.")