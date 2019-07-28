from __future__ import print_function
import os.path

import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar10
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from grda import GRDA
#from ngrda import NGRDA
import pandas as pd

batch_size = 100
nb_classes = 10
nb_epoch = 100

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40 # the standard densenet 40-12 setting, the smallest model in the original paper
nb_dense_block = 3
growth_rate = 12
nb_filter = -1
dropout_rate = 0.0 # 0.0 for data augmentation

model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                          growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None)
print("Model created")
allcounts = np.sum([x.size for x in model.get_weights()])
print(allcounts)

model.summary()
optimizer = GRDA(lr=0.005, c=0.001, mu=0.5 ) 
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX = densenet.preprocess_input(trainX)
testX = densenet.preprocess_input(testX)

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)


generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0)


results_acc = []
result_acc= []
results_loss = []
result_loss = []
test_acc_results = []
test_loss_results = []
nonzero_weights = []

for i in range (1):
    result_acc_e = []
    result_loss_e = []
    test_acc_e = []
    test_loss_e = []
    test_loss_e = []
    nonzero_weight_e = []
    for j in range(nb_epoch):
         history = model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(trainX) // batch_size, epochs=1,
                    validation_data=(testX, Y_test),
                    validation_steps=testX.shape[0] // batch_size, verbose=1)
                        #callbacks=callbacks,
         nzcounts = np.sum([np.count_nonzero(x) for x in model.get_weights()])
         print(nzcounts / allcounts, nzcounts, allcounts)
         if j % 2 == 0:
             test_loss_j, test_acc_j = model.evaluate(testX, Y_test)
             test_acc_e.append(test_acc_j)
             test_loss_e.append(test_loss_j)
             nonzero_weight_e.append(np.sum([np.count_nonzero(x) for x in model.get_weights()]) / allcounts)
         result_acc_e.append(history.history['acc'][0])
         result_loss_e.append(history.history['loss'][0])
    test_loss.append(test_loss_e)
    test_acc.append(test_acc_e)
    result_acc.append(result_acc_e)
    result_loss.append(result_loss_e)
    nonzero_weight.append(nonzero_weight_e)
    print("##### NEW OPTIMIZER #####")
    print(np.mean(result_acc, axis=0))
    print(np.mean(result_loss, axis=0))
    print(np.mean(test_acc, axis=0))
    print(np.mean(test_loss, axis=0))
    print(nonzero_weight)
    results_acc.append(np.mean(result_acc,axis=0))
    results_loss.append(np.mean(result_loss,axis=0))
    test_acc_results.append(np.mean(test_acc,axis=0))
    test_loss_results.append(np.mean(test_loss,axis=0))
    nonzero_weights.append(np.mean(nonzero_weight,axis=0))

df = pd.DataFrame(results_acc)
df.to_csv("cifar_acc_train_densenet40-12.csv")
df = pd.DataFrame(results_loss)
df.to_csv("cifar_loss_train_densenet40-12.csv")
df = pd.DataFrame(test_acc_results)
df.to_csv("cifar_acc_test_densenet40-12.csv")
df = pd.DataFrame(test_loss_results)
df.to_csv("cifar_loss_test_densenet40-12.csv")
df = pd.DataFrame(nonzero_weights)
df.to_csv("nonzero_weights_densenet40-12.csv")

# save weights
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'Cifar10_densenet40-12.h5'

model_json = model.to_json()
with open("model_Cifar10_densenet40-12.json", "w") as json_file:
    json_file.write(model_json)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
