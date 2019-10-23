import numpy as np
import cv2 # openCV Python interface
import math


def prewitt_filter(image):
    """Implementation of the Prewitt-filter

        It convolves 2 kernels on the image to appriximate the gradient in
        the horizontal and vertical direction, and then returns the gradient
        magnitude and orientation of the gradient normal to detect edges on the image"""

    offset = 1
    height, width = image.shape

    prewitt_mtx = 1./3. * np.array([[-1, 0, 1]] * 3)

    #vectorize for faster computation
    prewitt_horizontal = prewitt_mtx.ravel()
    prewitt_vertical = prewitt_mtx.T.ravel()

    grad_x = np.zeros(image.shape)
    grad_y = np.zeros(image.shape)
    for i in range(offset, height-offset):
        for j in range(offset, width-offset):
            box = image[i-offset:i+offset+1, j-offset:j+offset+1]
            flattened_box = box.ravel()

            # dot product with the flattened version
            grad_x[i, j] = flattened_box.dot(prewitt_horizontal)
            grad_y[i, j] = flattened_box.dot(prewitt_vertical)

    grad_magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))

    grad_orientation = np.arctan(grad_x/grad_y)

    #  cutting the offset
    return grad_magnitude[offset:-offset, offset:-offset], grad_orientation[offset:-offset, offset:-offset]


def get_gradient_dir(x):
    """ Get gradient direction as array index shifts"""
    if x > 3/8 * np.pi:
        return 0, 1
    elif x > 1/8 * np.pi:
        return 1, 1
    elif x > -1/8 * np.pi:
        return 1, 0
    elif x > -3/8 * np.pi:
        return 1, -1
    else:
        return 0, 1


def non_maxima_supression(image, grad_magnitude, grad_orientation):
    """Implementation of Non-Maxima Supression

        To make wide edges thinner, this algorithm checks if the gradient
        magnitude is a local maxima compared to its neighbours along the
        gradient direction. If not, it sets them to zero. The result is an
        image with thin edges."""
    grad_magnitude = np.pad(grad_magnitude, ((1, 1), (1, 1)))
    grad_orientation = np.pad(grad_orientation, ((1, 1), (1, 1)))

    offset = 1
    height, width = image.shape
    for i in range(offset, height-offset):
        for j in range(offset, width-offset):
            x_dir, y_dir = get_gradient_dir(grad_orientation[i, j])
            curr_grad = grad_magnitude[i, j]

            #set to 0 where intensity is not local maxima
            if grad_magnitude[i+x_dir, j+y_dir] > curr_grad or grad_magnitude[i-x_dir, j-y_dir] > curr_grad:
                grad_magnitude[i, j] = 0
    return grad_magnitude


im = cv2.imread("images/natasha.png", cv2.IMREAD_GRAYSCALE)
grad_magnitude, grad_orientation= prewitt_filter(im)

supressed_edges = non_maxima_supression(im, grad_magnitude, grad_orientation)

cv2.imshow("Original Image", im)
cv2.waitKey(0)
cv2.imshow("Perwitt result", grad_magnitude.astype(np.uint8))
cv2.waitKey(0)
cv2.imshow("Non-maxima supression result", supressed_edges.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()