from __future__ import print_function
import plaidml.keras
plaidml.keras.install_backend()
import keras
import tensorflow as tf
from keras.datasets import cifar10
from keras.callbacks import LambdaCallback  # for printing weights
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from grda import GRDA
import numpy as np
import pandas as pd

batch_size = 10
num_classes = 10
epochs = 20
data_augmentation = False
num_predictions = 50


# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

## same architecture as the Keras CNN sample code
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

results_acc = []
result_acc= []
results_loss = []
result_loss = []
test_acc_results = []
test_loss_results = []
l2= [GRDA(lr=0.005, c=0.001, mu=0.5)]
print_weights = LambdaCallback(on_epoch_begin=lambda batch, logs: print(model.layers[0].get_weights()))

allcounts = np.sum([x.size for x in model.get_weights()])

for opt in l2:

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    #model.save_weights('initial_weights2_K_cnn.h5')
    #model.load_weights('initial_weights2.h5')
    initial_weights = model.get_weights()
    result_acc = []
    result_loss = []
    test_loss = []
    test_acc = []
    nonzero_weights = []
    for i in range (2):
        #model.set_weights(initial_weights)
        result_acc_e = []
        result_loss_e = []
        test_acc_e = []
        test_loss_e = []
        test_loss_e = []
        nonzero_weights_e = []
        for j in range (20):
            print("nonzero ratio", "nonzero count", "all weights count")
            nzcounts = np.sum([np.count_nonzero(x) for x in model.get_weights()])
            print(nzcounts / allcounts, nzcounts, allcounts)
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1) # history = model.fit(x_train, y_train,batch_size=batch_size,epochs=1,verbose=1, callbacks=[print_weights])
            if j % 2 == 0 :
                test_loss_j, test_acc_j = model.evaluate(x_test, y_test)
                test_acc_e.append(test_acc_j)
                test_loss_e.append(test_loss_j)
                nonzero_weights_e.append(np.sum([np.count_nonzero(x) for x in model.get_weights()]) / allcounts)
            result_acc_e.append(history.history['acc'][0])
            result_loss_e.append(history.history['loss'][0])
        test_loss.append(test_loss_e)
        test_acc.append(test_acc_e)
        result_acc.append(result_acc_e)
        result_loss.append(result_loss_e)
        nonzero_weights.append(nonzero_weights_e)
    print("##### NEW OPTIMIZER #####")
    print(np.mean(result_acc,axis=0))
    print(np.mean(result_loss,axis=0))
    print(np.mean(test_acc,axis=0))
    print(np.mean(test_loss,axis=0))
    print(nonzero_weights)
    results_acc.append(np.mean(result_acc,axis=0))
    results_loss.append(np.mean(result_loss,axis=0))
    test_acc_results.append(np.mean(test_acc,axis=0))
    test_loss_results.append(np.mean(test_loss,axis=0))


df = pd.DataFrame(results_acc)
df.to_csv("results/cifar_acc_train_cnn.csv")
df = pd.DataFrame(results_loss)
df.to_csv("results/cifar_loss_train_cnn.csv")
df = pd.DataFrame(test_acc_results)
df.to_csv("results/cifar_acc_test_cnn.csv")
df = pd.DataFrame(test_loss_results)
df.to_csv("results/cifar_loss_test_cnn.csv")
df = pd.DataFrame(nonzero_weights)
df.to_csv("results/nonzero_weights_cnn.csv")


