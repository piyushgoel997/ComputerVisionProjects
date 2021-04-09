import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Ref - https://www.tensorflow.org/tutorials/keras/classification
# Ref - https://keras.io/examples/vision/mnist_convnet/

# B set random seed to make the code repeatable
np.random.seed(42)

# A load the data set
mnist_dataset = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# A look at 4 example images from the dataset
plt.figure(figsize=(1, 1))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap='gray')
    plt.xlabel(train_labels[i])
plt.savefig("task-1a")
plt.show()

# scaling the values to be between 0 and 1
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

# reshape each image to 28, 28, 1 to apply convolution layers
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)
print("train shape", train_images.shape)
print("test shape", test_images.shape)

# convert output to one-hot
num_labels = 10
train_labels = tf.keras.utils.to_categorical(train_labels, num_labels)
test_labels = tf.keras.utils.to_categorical(test_labels, num_labels)

# C make the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),  # Conv Layer 32 3x3 filters
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3)),  # Conv Layer 32 3x3 filters
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),  # Max pooling Layer 2x2
    tf.keras.layers.Dropout(rate=0.25),  # Dropout layer with 0.25 rate
    tf.keras.layers.Flatten(),  # Flatten Layer
    tf.keras.layers.Dense(units=128, activation='relu'),  # Dense Layer 128, relu activation
    tf.keras.layers.Dropout(rate=0.5),  # Dropout Layer with 0.5
    tf.keras.layers.Dense(units=10, activation='softmax')  # Dense Layer 10 for output with softmax
])

# C Compile model with Loss fn -> Categorical cross-entropy, Optimizer -> Adam, Metric -> Accuracy
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# D train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# print model summary
model.summary()

# D Plot of training and testing accuracy after each epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Curve')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.savefig("task-1d")
plt.show()

# E save model to a file
model.save("my_model.h5")

# load the model
loaded_model = tf.keras.models.load_model("my_model.h5")

# This function prints the predicted class, the class probabilities (or the model output) and the correct classes.
def print_predictions(mod, test_imgs, lbls):
    for p, c, l in zip(mod.predict(test_imgs), mod.predict_classes(test_imgs), lbls):
        pr = [round(x, 2) for x in p]
        print("The predicted digit is", c, "with the class probabilities", pr, "and the correct digit", np.argmax(l))


# E print the predictions for the first 10 test examples
print_predictions(loaded_model, test_images[:10], test_labels[:10])

# F load the handwritten test files
test_files = []
for i in range(10):
    img = cv2.imread("test_files/" + str(i) + ".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = np.array(img).astype("float32") / 255
    test_files.append(img)
test_files = np.array(test_files)
test_files = np.expand_dims(test_files, -1)

# F create the labels for the handwritten test files
test_files_labels = list(range(10))
test_files_labels = tf.keras.utils.to_categorical(test_files_labels, num_labels)

# F evaluate the model on the handwritten test images
loaded_model.evaluate(test_files, test_files_labels)
print_predictions(loaded_model, test_files, test_files_labels)