import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(42)


# This is a initializer that can be used to initialize layers with the gabor filters.
def gabor_initializer(shape, dtype=None):
    # ref: http://amroamroamro.github.io/mexopencv/opencv/gabor_demo.html
    thetas = [i * np.pi / shape[-1] for i in range(shape[-1])]
    kernels = np.zeros(shape)
    for i, t in enumerate(thetas):
        kernels[:, :, 0, i] = cv2.getGaborKernel(shape[:2], 4, t, 10, 0.5)
    kernels = np.array(kernels).astype("float32").reshape(shape)
    return tf.convert_to_tensor(kernels)


# train and test the model on the mnist dataset
def train_test_model(initializer, name, load_model=False):
    # load data
    mnist_dataset = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()
    # scaling the values to be between 0 and 1
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255
    # reshape each image to 28, 28, 1 to apply convolution layers
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)
    # convert output to one-hot
    num_labels = 10
    train_labels = tf.keras.utils.to_categorical(train_labels, num_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_labels)

    if not load_model:
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(28, 28, 1)),
            # un-trainable layer with gabor filters
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), trainable=False, kernel_initializer=initializer),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),  # Conv Layer 32 3x3 filters
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),  # Max pooling Layer 2x2
            tf.keras.layers.Dropout(rate=0.25),  # Dropout layer with 0.25 rate
            tf.keras.layers.Flatten(),  # Flatten Layer
            tf.keras.layers.Dense(units=128, activation='relu'),  # Dense Layer 128, relu activation
            tf.keras.layers.Dropout(rate=0.5),  # Dropout Layer with 0.5
            tf.keras.layers.Dense(units=10, activation='softmax')  # Dense Layer 10 for output with softmax
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # train the model
        history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
        model.summary()

        # Plot of training and testing accuracy after each epoch
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Accuracy Curve')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'])
        plt.savefig(str(name) + "_acc")
        plt.show()

        # save model to a file
        model.save(str(name) + ".h5")
    else:
        model = tf.keras.models.load_model(str(name) + ".h5", custom_objects={"gabor_initializer": gabor_initializer})

    # load the handwritten test files
    test_files = []
    for i in range(10):
        img = cv2.imread("../task1/test_files/" + str(i) + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = np.array(img).astype("float32") / 255
        test_files.append(img)
    test_files = np.array(test_files)
    test_files = np.expand_dims(test_files, -1)

    # create the labels for the handwritten test files
    test_files_labels = list(range(10))
    test_files_labels = tf.keras.utils.to_categorical(test_files_labels, num_labels)

    # evaluate the model on the handwritten test images
    model.evaluate(test_files, test_files_labels)
    print(model.predict_classes(test_files))


train_test_model(gabor_initializer, "gabor_model", load_model=True)
# test accuracy = 0.9
# 10/10 [==============================] - 0s 11ms/sample - loss: 0.9860 - accuracy: 0.9000
# [0 1 2 3 4 5 5 7 8 9]
