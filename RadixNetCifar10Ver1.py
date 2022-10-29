import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, Flatten, GlobalAveragePooling2D, ReLU
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data,
                             test_labels) = tf.keras.datasets.cifar10.load_data()

train_data = train_data.astype(np.float32)/255.0
test_data = test_data.astype(np.float32)/255.0
train_cat_labels = to_categorical(train_labels)
test_cat_labels = to_categorical(test_labels)

batchSize = 128


class CNNModel(tf.keras.Model):

    def __init__(self, N, D):
        super().__init__()

        self.connectivity = self.createConnectivity(N, D)

        N_ = 1
        for n in N[0]:
            N_ = N_*n

        self.nodes = np.array(D)*N_
        self.connectivity.insert(0, np.ones((self.nodes[0], 3)))

        self.convlayers = []
        for l in range(len(self.nodes)):
            layer = []

            for k in range(self.nodes[l]):
                layer.append(Conv2D(filters=1, kernel_size=(
                    3, 3), padding='same', activation='relu'))

            self.convlayers.append(layer)

        self.dense1 = Dense(units=50, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def createConnectivity(self, N, D):

        connectivity = []
        W = []
        N_ = 1
        for n in N[0]:
            N_ = N_*n

        I = np.eye(N_)
        P = np.roll(I, axis=0, shift=1)

        for n in N:

            v = 1
            for n_ in n:

                w = np.zeros(N_)
                for k in range(n_):
                    if (k*v) == 0:
                        tmp = I
                    else:
                        tmp = np.roll(I, axis=0, shift=k*v)
                    w = w + tmp
                W.append(w)

                v = v * n_

        for i in range(len(W)):
            w_ = np.ones((D[i+1], D[i]))

            connectivity.append(np.kron(w_, W[i]))

        return connectivity

    def call(self, inputs):

        i = tf.reshape(inputs, (-1, 32, 32, 3))

        prevmaps = []

        prevmaps.append(tf.expand_dims(i[:, :, :, 0], axis=3))

        prevmaps.append(tf.expand_dims(i[:, :, :, 1], axis=3))
        prevmaps.append(tf.expand_dims(i[:, :, :, 2], axis=3))
        for l in range(len(self.nodes)):

            fmaps = []
            for k in range(self.nodes[l]):

                indices = np.nonzero(self.connectivity[l][k])[0]
                x = tf.concat([prevmaps[i] for i in indices], axis=3)

                fmaps.append(self.convlayers[l][k](x))

            prevmaps = fmaps

        x = tf.concat(fmaps, axis=3)
        x = Flatten()(x)
        x = self.dense1(x)
        y = self.dense2(x)

        return y


cnn_model = CNNModel(N=[[3, 3], [3, 3], [9]], D=[1, 2, 3, 4, 5, 6])
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(x=train_data, y=train_cat_labels, epochs=100,
              batch_size=batchSize, validation_data=(test_data, test_cat_labels))
