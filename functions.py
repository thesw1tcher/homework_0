from math import sqrt

def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    ans = 1
    for el in [x[i][i] for i in range(min(len(x), len(x[0])))]:
        if el:
            ans *= el
    return ans



def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """
    x.sort()
    y.sort()
    return x == y


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """
    ans = 0
    fl = 1
    for i in range(1, len(x)):
        if x[i - 1] == 0 and (fl or x[i] > ans):
            fl = 0
            ans = x[i]
    return ans


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """
    ans = [[img[i][j][0] * coefs[0] + img[i][j][1] * coefs[1] + img[i][j][2] * coefs[2] for j in range(len(img[0]))] for
           i in range(len(img))]
    return ans


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """
    cur = x[0]
    l = 0
    elements = []
    counters = []
    for el in x:
        if el == cur:
            l += 1
        else:
            elements.append(cur)
            counters.append(l)
            cur = el
            l = 1
    elements.append(cur)
    counters.append(l)
    return elements, counters


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """
    return [[sqrt(sum([(e[i]-el[i])**2 for i in range(len(e))])) for e in y] for el in x]

