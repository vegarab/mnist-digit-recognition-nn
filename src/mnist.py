import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


def preprocess_dataset():
    """Gather and process mnist-dataset"""
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

def setup_train_network(train_images, train_labels, test_images, test_labels):
    """Setup neural network layers and fit the model"""
    network = models.Sequential()

    network.add(layers.Flatten(input_shape=train_images.shape[1:]))
    network.add(layers.Dense(196, activation='relu'))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    network.summary()
    network.fit(train_images, 
                train_labels, 
                epochs=8, 
                batch_size=128, 
                validation_split = 0.25)

    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)

    ### ---- VISUALIZATION ---- ###
    history = network.history.history

    # plot losses
    plt.figure(figsize=(12, 8))
    plt.plot(history["val_loss"], label="Validation loss")
    plt.plot(history["loss"], label="Training loss")
    plt.show()

    # plot accuracy
    plt.figure(figsize=(12, 8))
    plt.plot(history["val_acc"], label="Validation accuracy")
    plt.plot(history["acc"], label="Training accuracy")
    plt.show()

    final_loss, final_accuracy = network.evaluate(test_images, test_labels)
    print("The final loss on the test set is:", final_loss)
    print("The final accuracy on the test set is:", final_accuracy)

    # plot final losses
    plt.figure(figsize=(12, 8))
    plt.plot(history["val_loss"], label="Validation loss")
    plt.plot(history["loss"], label="Training loss")
    plt.plot([7], [final_loss], 'o', label="Final test loss")
    plt.show()


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = preprocess_dataset()
    history = setup_train_network(train_images, train_labels, 
                test_images, test_labels)
