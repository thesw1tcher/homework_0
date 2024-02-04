import numpy as np

def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """
    a = np.diagonal(x)
    return np.prod(np.where(a > 0, a, 1))


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """
    return np.sort(x) == np.sort(y)


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """
    a = np.concatenate((np.array([1]), x))[:-1]
    return max(np.where(a == 0, x, 0))


def f(i, j, img, coefs):
    # print(x)
    return img[i, j, 0] * coefs[0] + img[i, j, 1] * coefs[1] + img[i, j, 2] * coefs[2]


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """
    ans = np.fromfunction(lambda i, j: f(i, j, img, coefs), np.shape(img)[:-1], dtype=int)
    return ans


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """
    n = len(x)
    y = x[1:] != x[:-1]
    i = np.append(np.where(y), n - 1)
    z = np.diff(np.append(-1, i))
    return x[i], z


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """
    n = np.shape(x)[0]
    m = np.shape(y)[0]
    k = np.shape(x)[1]

    a = x[np.repeat(np.arange(n), m)]
    b = y[np.ravel(np.fromfunction(lambda i, j: j, (n, m), dtype=int))]
    c = np.reshape(np.sqrt(np.sum((a - b) ** 2, axis=1)), (n, m))
    return c
