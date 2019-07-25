# gRDA-Optimizer

Generalized Regularized Dual Averaging

## Requirements
    Keras version > 2.9
    Tensorflow version

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

opt = GRDA(learning_rate = 0.005, c = 0.005, mu = 0.51)
opt_r = opt.minimize(R_loss, var_list = r_vars)
sess.run([R_loss, opt_r], feed_dict = {data: train_x, y: train_y})
```
