import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.callbacks import CSVLogger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def preprocess_dataset():
    ''' Gather and process mnist-dataset '''
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # normalize dataset using z-score
    pixel_mean = train_images.mean(axis=0)
    pixel_std = train_images.std(axis=0) + 1e-10
    train_images = (train_images - pixel_mean) / pixel_std
    test_images = (test_images - pixel_mean) / pixel_std

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # reshape (60000, 28, 28) to (60000, 28, 28, 1)
    train_images = train_images[:,:,:, np.newaxis].astype(np.float32)
    test_images = test_images[:,:,:, np.newaxis].astype(np.float32)

    return train_images, train_labels, test_images, test_labels

def simple_network(train_images, train_labels, test_images, test_labels):
    ''' Setup neural network layers and fit the model, 
        Simplest possible network to achieve 95% accuracy '''
    network = models.Sequential()

    network.add(layers.Flatten(input_shape=train_images.shape[1:]))
    network.add(layers.Dense(32, activation='relu'))
    network.add(layers.Dropout(0.25))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    epochs = 15
    network, history = fit_save_network(network, epochs, 'simple')
    visualization(network, history, epochs)

def fit_save_network(network, epochs, name=None):
    ''' Fit and save a network, or load from disk '''
    print()
    network.summary()
    if not os.path.isfile('{}.log'.format(str(name))) or not os.path.isfile('{}.h5'.format(str(name))):
        csv_logger = CSVLogger('{}.log'.format(str(name)), 
                               separator=',', 
                               append=False)
        network.fit(train_images, train_labels,
                  batch_size=128,
                  epochs=epochs,
                  validation_split = 0.25,
                  callbacks = [csv_logger])
        # serialize weights to HDF5
        network.save_weights('{}.h5'.format(str(name)))
        print('Saved model to disk')
        history = network.history.history
    else:
        log_data = pd.read_csv('{}.log'.format(str(name)), 
                               sep=',', 
                               engine='python')
        network.load_weights('{}.h5'.format(str(name)))
        print('Loaded model from disk')
        history = log_data

    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)

    return network, history


def best_network(train_images, train_labels, test_images, test_labels):
    ''' Setup neural network layers and fit the model, 
        Simplest possible network to achieve 95% accuracy '''
    network = models.Sequential()

    network.add(layers.Flatten(input_shape=train_images.shape[1:]))
    network.add(layers.Dense(25, activation='relu'))
    network.add(layers.Dropout(0.25))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])


def visualization(network, history, epochs):
    ''' Visulize loss and acurracy for a network '''
    # plot losses
    plt.figure(figsize=(12, epochs))
    plt.plot(history['val_loss'], label='Validation loss')
    plt.plot(history['loss'], label='Training loss')
    plt.legend()
    plt.show()

    # plot accuracy
    plt.figure(figsize=(12, epochs))
    plt.plot(history['val_acc'], label='Validation accuracy')
    plt.plot(history['acc'], label='Training accuracy')
    plt.legend()
    plt.show()

    final_loss, final_accuracy = network.evaluate(test_images, test_labels)
    print('The final loss on the test set is:', final_loss)
    print('The final accuracy on the test set is:', final_accuracy)

    # plot final losses
    plt.figure(figsize=(12, epochs))
    plt.plot(history['val_loss'], label='Validation loss')
    plt.plot(history['loss'], label='Training loss')
    plt.plot([epochs-1], [final_loss], 'o', label='Final test loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = preprocess_dataset()
    simple_network(train_images, train_labels, test_images, test_labels)
    best_network(train_images, train_labels, test_images, test_labels)
