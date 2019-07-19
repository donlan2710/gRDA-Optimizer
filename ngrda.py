from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K
import tensorflow as tf

class NGRDA(Optimizer):
    """NGRDA optimizer.
    """

    def __init__(self, lr=0.01, c=0., mu=0.7, delta=0.01, **kwargs):
        super(NGRDA, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr') # lr
            self.mu = K.variable(mu, name='mu') # mu
            self.delta = K.variable(delta, name="delta")
            self.c = K.variable(c, name='c') # c

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        # how to get the initializer of params?
        accumulators = [K.random_uniform_variable(shape, low=-0.1, high=0.1, seed=123) for shape in shapes]
        theta = [K.random_uniform_variable(shape, low=-0.1, high=0.1, seed=456) for shape in shapes]
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        self.weights = [self.iterations] + theta + accumulators
        mu = self.mu
        c = self.c
        delta = self.delta
        l1 = c * K.pow(lr, 0.5 + mu) * K.pow(K.cast(self.iterations, K.floatx()), mu)

        ss = 0
        for t, g in zip(theta, grads):
            ss += K.sum(t * g)

        print(ss)

        for p, g, a, t in zip(params, grads, accumulators, theta):
            new_t = (1 - lr) * t + delta * g - delta * (1 - lr) * ss * g
            self.updates.append(K.update(t, new_t))
            new_a = a - lr * new_t  # Gradient Step
            self.updates.append(K.update(a, new_a))
            new_p = tf.sign(new_a) * tf.maximum(abs(new_a) - l1, 0)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'mu': float(K.get_value(self.mu)),
                  'c': float(K.get_value(self.c)),
                  'delta': float(K.get_value(self.delta))
                  }
        base_config = super(NGRDA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
