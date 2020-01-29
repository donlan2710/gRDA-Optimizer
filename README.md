# gRDA-Optimizer

"Generalized Regularized Dual Averaging" is an optimizer that can learn a small sub-network during training, if one starts from an overparameterized dense network. 

#### Citation: Chao, S.-K. and Cheng, G. (2019). gRDA and its dynamics. https://arxiv.org/pdf/1909.10072.pdf

Here is an illustration of the optimizer using the simple 6-layer CNN https://keras.io/examples/cifar10_cnn/. The experiments are done using lr = 0.005 for SGD, SGD momentum and gRDAs. c = 0.005 for gRDA. lr = 0.005 and 0.001 for Adagrad and Adam, respectively.

<img src = 'https://github.com/donlan2710/gRDA-Optimizer/blob/master/pics/cifar_cnn_acc_test_multiopt.png' width=46%/> <img src = 'https://github.com/donlan2710/gRDA-Optimizer/blob/master/pics/cifar_cnn_nonzero_weights_multiopt.png' width=46%/>

## Update

09/25/2019: A bug with the initializer in ```grda.py```, ```grda_plaidml.py``` and ```grda_pytorch.py``` is fixed.

08/07/2019: A bug in ```grda_tensorflow.py``` is fixed.

## Requirements
    Keras version >= 2.2.5
    Tensorflow version >= 1.14.0

## How to use

There are three hyperparameters: Learning rate (lr), sparsity control mu (mu), and initial sparse control constant (c) in gRDA optimizer.

* lr: as a rule of thumb, use the learning rate for SGD. Scale the learning rate with the batch size.
* mu: 0.5 < mu < 1. Greater mu will make the parameters more sparse. For large tasks, e.g. ImageNet, mu can set close to 0.5, e.g. 0.501. For small task, e.g. CIFAR-10, mu can be 0.7.
* c: a small number, e.g. 0 < c < 0.05. This usually has small effect on the performance.

### Keras

Suppose the loss function is the categorical crossentropy,

``` python
from grda import GRDA

opt = GRDA(lr = 0.005, c = 0.005, mu = 0.7)
model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=['accuracy'])
```

### Tensorflow
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
### PyTorch
You can check the test file `mnist_test_pytorch.py`. 
The essential part is below.
``` python
from grda_pytorch import gRDA

optimizer = gRDA(model.parameters(), lr=0.005, c=0.1, mu=0.5)
# loss.backward()
# optimizer.step()
```

See ```mnist_test_pytorch.py``` for an illustration on customized learning rate schedule.

### PlaidML 

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
Then the optimizer can be used in the same way as Keras.
