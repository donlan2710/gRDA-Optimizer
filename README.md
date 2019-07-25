# gRDA-Optimizer

Generalized Regularized Dual Averaging

## Requirements
    Keras version >= 2.9
    Tensorflow version >= 1.14.0

## How to use

Best Options for learning rate (lr), mu (mu), and smoothing constant (c) in gRDA optimizer  
    lr: 0 < lr < 0.05  
    mu: 0 < mu < 1  
    c: 0 < c < 0.05  

### With Keras

Suppose the loss function is the categorical crossentropy,

``` python
from grda import GRDA

opt = GRDA()
model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=['accuracy'])
```

### With Tensorflow
``` python
from grda_tensorflow import GRDA

n_epochs = 20
batch_size = 10
batches = 50000/batch_size # CIFAR-10 number of minibatches

opt = GRDA(learning_rate = 0.005, c = 0.005, mu = 0.51)
opt_r = opt.minimize(R_loss, var_list = r_vars)
with tf.Session(config=session_conf) as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(n_epochs + 1):
        for b in range(batches):
            sess.run([R_loss, opt_r], feed_dict = {data: train_x, y: train_y})
```
