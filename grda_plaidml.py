from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K
class GRDA(Optimizer):
    """GRDA optimizer.
    """

    def __init__(self, lr=0.01, c=0., mu=0.7, **kwargs):
        super(GRDA, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr') # lr
            self.mu = K.variable(mu, name='mu') # mu
            if c==0:
                raise ValueError("c = 0 is equivalent to SGD. Please use SGD.")
            self.c = K.variable(c, name='c') # c
            print("lr = ", lr, ", c=", c, ", mu = ", mu)

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.random_uniform_variable(shape, low=-0.1, high=0.1, seed=1) for shape in shapes]
        self.updates = [K.update_add(self.iterations, 1)]


        lr = self.lr
        mu = self.mu
        c = self.c
        l1 = c * K.pow(lr, 0.5 + mu) * K.pow(K.cast(self.iterations, K.floatx()) + 1, mu)
        for p, g, a in zip(params, grads, accumulators):
            new_a = a - lr * g 
            self.updates.append(K.update(a, new_a))
            new_p = K.softthreshold(new_a, l1)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'mu': float(K.get_value(self.mu)),
                  'c': float(K.get_value(self.c))
                  }
        base_config = super(GRDA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
