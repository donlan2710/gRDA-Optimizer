# gRDA-Optimizer

Generalized Regularized Dual Averaging. Paper preprint is coming soon.

<img src = 'https://github.com/donlan2710/gRDA-Optimizer/blob/master/pics/cifar_cnn_acc_test_multiopt.png' width=40%/> <img src = 'https://github.com/donlan2710/gRDA-Optimizer/blob/master/pics/cifar_cnn_nonzero_weights_multiopt.png' width=40%/>

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

### With PlaidML 

Be cautious that it can be unstable with Mac when GPU is implemented, see https://github.com/plaidml/plaidml/issues/168. 

To run, define the softthreshold function in the plaidml backend file (plaidml/keras):

```python
def softthreshold(x, t):
     x = clip(x, -t, t) * (builtins.abs(x) - t) / t
     return x
```

In the main file, add the following before importing other libraries

```python
import plaidml.keras
plaidml.keras.install_backend()

from grda_plaidml import GRDA
```
