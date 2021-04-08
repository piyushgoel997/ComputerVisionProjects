import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_filters_and_filtered_output(model, images, layer_number=0, example_number=0):
    # A get the weights of the first layer of the model
    first_layer_wts = model.get_layer(index=layer_number).weights[0]

    # B plotting the filters
    # Ref: https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
    # yellow means more positive values, and blue means more negative value
    fig, axs = plt.subplots(8, 4, figsize=(8, 16))
    for i in range(8):
        for j in range(4):
            wts = first_layer_wts[:, :, 0, 4 * i + j]
            print(wts)
            axs[i, j].imshow(wts)
    plt.show()

    # C Effect of filters on the first training example
    fig, axs = plt.subplots(8, 4, figsize=(8, 16))
    for i in range(8):
        for j in range(4):
            wts = first_layer_wts[:, :, 0, 4 * i + j]
            im = cv2.filter2D(images[example_number], 3, wts.numpy())
            axs[i, j].imshow(im)
    plt.show()


def build_and_apply_truncated_model(model, images, layer_number=0, example_number=0):
    # D building a truncated model and applying it to the first layer
    first_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=layer_number).output)
    img = images[example_number:example_number + 1]
    img = img.reshape(*img.shape, 1)
    pred = first_layer_model.predict(img)
    fig, axs = plt.subplots(8, 4, figsize=(8, 16))
    for i in range(8):
        for j in range(4):
            axs[i, j].imshow(pred[0, :, :, 4 * i + j])
    plt.show()


mnist_dataset = tf.keras.datasets.mnist
(train_images, _), _ = mnist_dataset.load_data()
loaded_model = tf.keras.models.load_model("../task1/my_model.h5")
plot_filters_and_filtered_output(loaded_model, train_images)
build_and_apply_truncated_model(loaded_model, train_images)
# E plot after adding in the second conv layer to the truncated model
build_and_apply_truncated_model(loaded_model, train_images, layer_number=1)
# E plot after adding in the pooling layer to the truncated model
build_and_apply_truncated_model(loaded_model, train_images, layer_number=2)
# E plot of pooling layer for some other digits
build_and_apply_truncated_model(loaded_model, train_images, layer_number=2, example_number=1)
build_and_apply_truncated_model(loaded_model, train_images, layer_number=2, example_number=2)
build_and_apply_truncated_model(loaded_model, train_images, layer_number=2, example_number=3)
build_and_apply_truncated_model(loaded_model, train_images, layer_number=2, example_number=4)
build_and_apply_truncated_model(loaded_model, train_images, layer_number=2, example_number=5)
