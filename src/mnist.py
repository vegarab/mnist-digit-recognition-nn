import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

def preprocess_dataset():
    """Gather and process mnist-dataset"""
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()


    pixel_mean = train_images.mean(axis=0)
    pixel_std = train_images.std(axis=0) + 1e-10

    train_images = (train_images - pixel_mean) / pixel_std
    test_images = (test_images - pixel_mean) / pixel_std

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels

def setup_train_network(train_images, train_labels, test_images, test_labels):
    """Setup neural network layers and fit the model"""
    network = models.Sequential()

    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    network.fit(train_images, train_labels, epochs=5, batch_size=128)

    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = preprocess_dataset()
    setup_network(train_images, train_labels, test_images, test_labels)
