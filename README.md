# gRDA-Optimizer

"Generalized Regularized Dual Averaging" is an optimizer that can learn a small sub-network during training, if one starts from an overparameterized dense network. 


Please cite the following publication when referring to gRDA:

Chao, S.-K., Wang, Z., Xing, Y. and Cheng, G. (2020). Directional pruning of deep neural networks. *NeurIPS 2020*. Available at: https://arxiv.org/abs/2006.09358
<p align="center">
<img src = 'https://github.com/donlan2710/gRDA-Optimizer/blob/master/pics/intro_cifar100_conn_thumb.png' width=46%/>
    The DP is the asymptotic directional pruning solution computed with gRDA.
</p>
Here is an illustration of the optimizer using the simple 6-layer CNN https://keras.io/examples/cifar10_cnn/. The experiments are done using lr = 0.005 for SGD, SGD momentum and gRDAs. c = 0.005 for gRDA. lr = 0.005 and 0.001 for Adagrad and Adam, respectively.

<img src = 'https://github.com/donlan2710/gRDA-Optimizer/blob/master/pics/cifar_cnn_acc_test_multiopt.png' width=46%/> <img src = 'https://github.com/donlan2710/gRDA-Optimizer/blob/master/pics/cifar_cnn_nonzero_weights_multiopt.png' width=46%/>

## Requirements
    Keras version >= 2.2.5
    Tensorflow version >= 1.14.0

## How to use

There are three hyperparameters: Learning rate (lr), sparsity control mu (mu), and initial sparse control constant (c) in gRDA optimizer.

* lr: as a rule of thumb, use the learning rate for SGD. Scale the learning rate with the batch size.
* mu: 0.5 < mu < 1. Greater mu will make the parameters more sparse. In order to maintain comparable accuracy with the original network, for large tasks e.g. ImageNet, mu can set close to 0.5, e.g. 0.501. For small tasks, e.g. CIFAR-10, mu can be larger, e.g. 0.6. 
* c: a small number, e.g. 0 < c < 0.005. Greater c causes the model to be more sparse, especially at the early stage of training. c usually has small effect on the late stage of training. The influence of c is smaller than the influence of mu.

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

### Experiments

#### ResNet-50 on ImageNet 

These PyTorch models are based on the official implementation: [https://github.com/pytorch/examples/blob/master/imagenet/main.py](https://github.com/pytorch/examples/blob/master/imagenet/main.py)

| lr schedule                                   | c     | mu    | epoch | sparsity | top1 accuracy | file size | link                                                                       |
|-----------------------------------------------|-------|-------|-------|----------|---------------|-----------|----------------------------------------------------------------------------|
| fix lr=0.1 (SGD, no momentum or weight decay) | /     | /     | 89    | /        | 68.71         | 98MB      | [link](https://drive.google.com/open?id=1PRkLaINIS14D3l553X1ncVBjqL5ua3Np) |
| fix lr=0.1                                    | 0.005 | 0.55  | 145   | 91.54    | 69.76         | 195MB     | [link](https://drive.google.com/open?id=1nzjT1dcZnagWdtkK14wHCDI54UrmphCH) |
| fix lr=0.1                                    | 0.005 | 0.51  | 136   | 87.38    | 70.35         | 195MB     | [link](https://drive.google.com/open?id=1PQ2C5kNOvl0NpDa-_h9YxlW0BD5pMylZ) |
| fix lr=0.1                                    | 0.005 | 0.501 | 145   | 86.03    | 70.60         | 195MB     | [link](https://drive.google.com/open?id=12o3hUHV5ffjcBkN5xos5qFG8365Wk2xl) |
| lr=0.1 (ep1-140) lr=0.01 (after ep140)        | 0.005 | 0.55  | 150   | 91.59    | 73.24         | 195MB     | [link](https://drive.google.com/open?id=1jBFmHmtsPsoIS5KjApjv8ohoJ5_RQqPx) |
| lr=0.1 (ep1-140) lr=0.01 (after ep140)        | 0.005 | 0.51  | 146   | 87.28    | 73.14         | 195MB     | [link](https://drive.google.com/open?id=1UlwjvFO-Oxl9VVV36k5UrWZXWhh2nvDN) |
| lr=0.1 (ep1-140) lr=0.01 (after ep140)        | 0.005 | 0.501 | 148   | 86.09    | 73.13         | 195MB     | [link](https://drive.google.com/open?id=1MQt6T7fc6SZlmMdGM6jOpaKHG6Ca5ulr) |
