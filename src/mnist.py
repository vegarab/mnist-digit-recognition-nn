import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np

import matplotlib.pyplot as plt
from collections import Counter

def preprocess_dataset():
    """Gather and process mnist-dataset"""
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # normalize data with z-score
    pixel_mean = train_images.mean(axis=0)
    pixel_std = train_images.std(axis=0) + 1e-10
    train_images = (train_images - pixel_mean) / pixel_std
    test_images = (test_images - pixel_mean) / pixel_std

    """# --------------------- # 

    from keras.preprocessing.image import ImageDataGenerator

    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(train_images[i], cmap='gray')
    # show the plot
    plt.show()

    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
    # convert from int to float
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    #datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    print(len(train_images))
    datagen = ImageDataGenerator(zca_whitening=True)
    datagen.fit(train_images, 'in_images')
    print(len(train_images))

    for image_batch, label_batch in datagen.flow(train_images, train_labels, batch_size=9):
        for i in range(0,9):
            plt.subplot(330 + 1 + i)
            plt.imshow(image_batch[i].reshape(28, 28), cmap="gray")
        plt.show()
        break

    # ---------------------- #"""

    # Reshape data from (28, 28) to (28, 28, 1) which is expected by tf
    print("Old shape:", train_images.shape)
    train_images = train_images[:,:,:, np.newaxis].astype(np.float32)
    test_images = test_images[:,:,:, np.newaxis].astype(np.float32)
    print("New shape:", train_images.shape)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    #return train_images, train_labels, test_images, test_labels

#def setup_train_network(train_images, train_labels, test_images, test_labels):
    """Setup neural network layers and fit the model"""

    network = models.Sequential()
    network.add(layers.Flatten(input_shape = train_images.shape[1:]))
    #network.add(layers.Dense(28*28))
    network.add(layers.Dense(128, activation='relu'))
    network.add(layers.Dense(10, activation='softmax'))
    
    learning_rate = 0.5
    network.compile(optimizer=keras.optimizers.SGD(lr=learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    #network.summary()
    
    from keras.preprocessing.image import ImageDataGenerator

    #gen = ImageDataGenerator(width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08)
    gen = ImageDataGenerator(zca_whitening=True)
    gen.fit(train_images)
    test_gen = ImageDataGenerator()

    train_generator = gen.flow(train_images, train_labels, batch_size=64)
    test_generator = test_gen.flow(test_images, test_labels, batch_size=64)

    network.fit_generator(train_generator, steps_per_epoch=60000//64, epochs=7, validation_data=test_generator, validation_steps=10000//64)

    #network.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('test_acc:', test_acc)


if __name__ == '__main__':
    preprocess_dataset()
    #train_images, train_labels, test_images, test_labels = preprocess_dataset()
    #setup_train_network(train_images, train_labels, test_images, test_labels)
