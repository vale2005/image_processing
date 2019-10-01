import numpy as np
import math
import cv2 # openCV Python interface
import time

class FilterSizeNotOddException(Exception):
    """Raised when the filter size is not odd"""
    pass


def simple_box(image, filter_size):
    """Naive implementation of the box filter algorithm

    It basically runs a filter size x filter size kernel
    through the image that calculates the average of the
    elements. The function also returns the time elapsed
    during computation"""
    start_time = time.time()
    if filter_size % 2 != 1:
        raise FilterSizeNotOddException

    height, width = image.shape
    result = np.zeros(image.shape)
    offset = math.floor(filter_size/2.)

    for i in range(offset, height-offset):
        for j in range(offset, width-offset):
            box = image[i-offset:i+offset+1, j-offset:j+offset+1]
            result[i, j] = box.mean()


    end_time = time.time()
    #  cutting the offset
    return result[offset:-offset, offset:-offset], end_time-start_time


def run_box(image, n):
    """Implementation of the running box filter

    This algorithm also runs an nxn kernel through the image,
    but it does not calculate every value from sketch. Instead,
    it goes row by row and updates the sum by removing the values
    of the first row and adding the values in the next row. The
    function also returns the time elapsed during computation"""
    start_time = time.time()
    if n % 2 != 1:
        raise FilterSizeNotOddException

    height, width = image.shape

    # output shape will be smaller according to the filter size
    result = np.empty((height-n+1, width-n+1))

    # sum of the first filter size values in each column
    colsums = image[0:n].sum(axis=0)
    for i in range(height-n+1):
        # convolve [1/filter_size, 1/filter_size, ..., 1/filter_size]
        # kernel over the column sums to get running average
        result[i] = np.convolve(colsums, np.ones(n)/n**2, 'valid')

        # need to check if out of bounds
        if i != height-n:
            # subtract first row of window
            colsums -= image[i]
            # add next row
            colsums += image[i+n]

    end_time = time.time()
    return result, end_time-start_time

im = cv2.imread("images/phobos.png", cv2.IMREAD_GRAYSCALE)
res1, exec_time1 = simple_box(im,9)
res2, exec_time2 = run_box(im,9)

print("Execution time of the simple box filter:", exec_time1, "seconds")
print("Exeution time of the running box filter:", exec_time2, "seconds")

cv2.imshow("Original Image", im)
cv2.waitKey(0)
cv2.imshow("Simple box result", res1.astype(np.uint8))
cv2.waitKey(0)
cv2.imshow("Running box result", res2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()