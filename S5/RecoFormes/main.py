import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import keras


def median_filter(image: np.ndarray) -> np.ndarray:
    res = image.copy()
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            res[i, j] = np.median(res[i - 1:i + 2, j - 1:j + 2])
    return res


def lena():
    # Read an image
    image1: np.ndarray = imageio.v3.imread('datasets/Lena_noisy.png')
    image2: np.ndarray = median_filter(image1)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 16))
    axes[0].imshow(image1)
    axes[1].imshow(image2)

    plt.show()


def wiki():
    image1: np.ndarray = imageio.v3.imread('datasets/wikipedia_lowcontrast.jpg')

    v_max = 255
    hist = np.zeros(v_max + 1)
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            hist[image1[i, j]] += 1

    hist_cum = np.zeros(v_max + 1)
    for val in range(v_max):
        hist_cum[val] = hist_cum[val - 1] + hist[val]

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(30, 25))
    axes[0].bar(x=list(range(v_max + 1)), height=hist, color="g")
    axes[1].bar(x=list(range(v_max + 1)), height=hist_cum, color="b")
    axes[0].set_title("Histogram")
    axes[1].set_title("Histogram cumulated")
    axes[1].set_xlabel("Grayscale")
    axes[1].set_ylabel("Frequency")
    axes[0].set_xlabel("Grayscale")
    axes[0].set_ylabel("Frequency")

    plt.show()

    hr = hist_cum / np.sum(hist)

    new_image = image1.copy()
    for i in range(1, image1.shape[0] - 1):
        for j in range(1, image1.shape[1] - 1):
            new_image[i, j] = hr[image1[i, j]] * v_max

    fig, axes = plt.subplots(1, 2, figsize=(16, 16))
    axes[0].imshow(image1)
    axes[1].imshow(new_image)

    plt.show()

    hist_tr = np.zeros(v_max + 1)
    for i in range(new_image.shape[0]):
        for j in range(new_image.shape[1]):
            hist_tr[new_image[i, j]] += 1

    hist_cum_tr = np.zeros(v_max + 1)
    for val in range(v_max):
        hist_cum_tr[val] = hist_cum_tr[val - 1] + hist_tr[val]

    fig, axes = plt.subplots(1, 2, figsize=(30, 25))
    axes[0].bar(x=list(range(v_max + 1)), height=hist_tr, color="g")
    axes[1].bar(x=list(range(v_max + 1)), height=hist_cum_tr, color="b")
    axes[0].set_title("Histogram")
    axes[1].set_title("Histogram cumulated")
    axes[1].set_xlabel("Grayscale")
    axes[1].set_ylabel("Frequency")
    axes[0].set_xlabel("Grayscale")
    axes[0].set_ylabel("Frequency")

    plt.show()

    slope = (new_image.shape[0] * new_image.shape[1]) / v_max

    fig, axes = plt.subplots(figsize=(25, 25))

    fig.suptitle("Check cumulated histogram")
    axes.bar(x=list(range(v_max + 1)), height=hist_cum_tr)
    axes.axline((0.0, 0.0), slope=slope, color="r")
    axes.set_xlabel("Grayscale")
    axes.set_ylabel("Cumulated Frequency")

    plt.show()


def encode_class(c: str) -> int:
    if c == "male":
        return 1
    elif c == "female":
        return 0
    else:
        return -1


def voice():
    data = pandas.read_csv('datasets/voice.csv')

    nb_entries = len(data.values)
    nb_variables = len(data.columns)
    classes = data["label"].unique().tolist()
    nb_classes = len(classes)

    data_x = data.values[:, :nb_variables - 1]
    data_x = data_x.astype(np.float64)
    data_y = data.values[:, nb_variables - 1]
    encoded_y = np.array([encode_class(y) for y in data_y])

    x_train, x_test, y_train, y_test = train_test_split(data_x, encoded_y, test_size=0.3, random_state=100)

    # Create the neural network
    # Blank sequential neural network
    model = keras.models.Sequential()

    # Add layers
    model.add(keras.layers.Dense(10, input_dim=nb_variables - 1, activation="sigmoid"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    # Train the neural network
    # Compile
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # fit
    model.fit(x_train, y_train, epochs=100, batch_size=10)

    # Evaluate
    # apply to test
            # -> prediction vs data
            # -> confusion matrix
# Predict the classes for the test set
y_pred = model.predict(x_test)
y_pred_classes = np.round(y_pred).astype(int).flatten()

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

conf_matrix
recall_m = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
recall_f = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])

precision_m = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
precision_f = conf_matrix[1, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1])

print(f"Recall female: {recall_f}")
print(f"Recall male: {recall_m}")
print(f"Precision female: {precision_f}")
print(f"Precision male: {precision_m}")
            # -> Accuracy
score = model.evaluate(x_test, y_test)
print(score)
