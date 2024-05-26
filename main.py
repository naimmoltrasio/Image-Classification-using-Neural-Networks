import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

training_images, test_images = training_images / 255, test_images / 255


def plot_images(images, labels, num_rows, num_cols):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = axes.flatten()
    for img, ax, lbl in zip(images, axes, labels):
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {lbl}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def model_train(x_train, x_test, y_train, y_test):
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'Test accuracy: {test_acc}')
    model.save('image_classifier.keras')


num_images = 20
t_images = training_images[:num_images]
t_labels = training_labels[:num_images]
plot_images(t_images, t_labels, num_rows=4, num_cols=5)

model_train(training_images, test_images, training_labels, test_labels)

model = models.load_model('image_classifier.keras')
