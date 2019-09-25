from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K
import tensorflow as tf

class GRDA(Optimizer):
    """GRDA optimizer.
    """

    def __init__(self, lr=0.01, c=0., mu=0.7, initweight=None, **kwargs):
        super(GRDA, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr') # lr
            self.mu = K.variable(mu, name='mu') # mu
            self.c = K.variable(c, name='c') # c
            self.initweight = initweight

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        accumulators = [K.variable(value = K.get_value(p), dtype='float32') for p in params]
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        self.weights = accumulators
        mu = self.mu
        c = self.c
        l1 = c * K.pow(lr, 0.5 + mu) * K.pow(K.cast(self.iterations, K.floatx()), mu)
        for p, g, a in zip(params, grads, accumulators):
            new_a = a - lr * g  # Gradient Step
            self.updates.append(K.update(a, new_a))
            new_a_l1 = K.abs(new_a) - l1
            new_p = tf.sign(new_a) * K.maximum(new_a_l1, K.zeros(K.int_shape(new_a)))

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'mu': float(K.get_value(self.mu)),
                  'c': float(K.get_value(self.c))
                  }
        base_config = super(GRDA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
