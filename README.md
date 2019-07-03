# gRDA-Optimizer

Stochastic Mirror Descent Optimizer

## Requirements
    Keras version
    Tensorflow version

## How to use
``` python
opt = gRDA()
model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=['accuracy'])
```
Best Options for learning rate (lr), mu (mu), and smoothing constant (c) in gRDA optimizer
    lr: 0 < lr < 0.05 
    mu: 0 < mu < 1
    c: 0 < c < 0.05
