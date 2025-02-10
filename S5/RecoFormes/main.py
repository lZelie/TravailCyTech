import imageio
import matplotlib.pyplot as plt
import numpy as np


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


wiki()
