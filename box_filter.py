import numpy as np
import math

class FilterSizeNotOddException(Exception):
    """Raised when the filter size is not odd"""
    pass


def simple_box(image, filter_size):
    """Naive implementation of the box filter algorithm

    It basically runs a filter size x filter size kernel
    through the image that calculates the average of the
    elements"""
    if filter_size % 2 != 1:
        raise FilterSizeNotOddException

    height, width = image.shape
    result = np.zeros(image.shape)
    offset = math.floor(filter_size/2.)

    for i in range(offset, height-offset):
        for j in range(offset, width-offset):
            box = image[i-offset:i+offset+1, j-offset:j+offset+1]
            result[i, j] = box.mean()

    #  cutting the offset
    return result[offset:-offset, offset:-offset]


def run_box(image, n):
    """Implementation of the running box filter

    This algorithm also runs an nxn kernel through the image,
    but it does not calculate every value from sketch. Instead,
    it goes row by row and updates the sum by removing the values
    of the first row and adding the values in the next row."""
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

    return result


im = np.arange(100).reshape(10,10)
im[8,8] = 10*88
print(im)
res1 = simple_box(im,3)
res2 = run_box(im,3)
print(res1)
print(res2)


img = np.random.randn(1000000).reshape(1000,1000)

res1 = simple_box(img, 11)
print(res1)
res2 = run_box(img, 11)
print(res2)
print(np.isclose(res1, res2).all())