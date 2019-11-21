import numpy as np
import cv2 # openCV Python interface
from matplotlib import pyplot as plt

def otsu_threshold(image):
    """Implementation of Otsu algorithm for binary thresholding of images

        The algorithm finds a threshold by checking the between class
        varience for each possible value and it selects the one that
        maximizes it."""
    num_pixels = image.size

    relative_frequencies = np.bincount(image.flatten(), minlength=256)/num_pixels
    mean = image.mean()

    c1_rel_freq, c1_mean, c2_mean = relative_frequencies[0], 0, 0

    between_class_variances = np.zeros(256)

    for i in range(1,255):
        prev_freq = c1_rel_freq
        c1_rel_freq += relative_frequencies[i]
        c1_mean = (prev_freq * c1_mean + i * relative_frequencies[i]) / c1_rel_freq if c1_rel_freq != 0 else 0

        c2_mean = (mean - c1_rel_freq*c1_mean)/(1.-c1_rel_freq) if 1-c1_rel_freq !=0 else 0

        between_class_variances[i] = c1_rel_freq * (1.-c1_rel_freq) * (c1_mean-c2_mean)**2

    optimal_threshold = np.argmax(between_class_variances)
    thresholded_image = np.where(image < optimal_threshold, 0, 255)

    return relative_frequencies, optimal_threshold, thresholded_image


im = cv2.imread("images/natasha.png", cv2.IMREAD_GRAYSCALE)
relative_frequencies, optimal_threshold, thresholded_image = otsu_threshold(im)


plt.plot(np.arange(256), relative_frequencies)
plt.axvline(optimal_threshold, c="black")
plt.show()

cv2.imshow("Original Image", im)
cv2.waitKey(0)
cv2.imshow("Thresholded image", thresholded_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
