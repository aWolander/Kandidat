import numpy as np
import random


def quantize(vec, q):
    abs_vec = np.abs(vec)
    x_max = np.max(abs_vec)
    x_min = np.min(abs_vec)
    # Could probably be made faster .vectorize() is essentially a for loop. At least part of quantizer
    # can be made into tensor operations
    quantizer = np.vectorize(Q_elem)
    return np.atleast_2d(quantizer(vec, q, x_max, x_min))


def Q_elem(elem, q, x_max, x_min):
    # WARNING! this breaks for tensors of identical elements as x_max - x_min = 0.
    # Maybe add 1e-6 to quotient
    return np.sign(elem) * (x_min + (x_max - x_min) *
                            phi((np.abs(elem) - x_min) / (x_max - x_min), q))


def phi(elem, q):
    for l_temp in range(q):
        if l_temp / q <= elem <= (l_temp + 1) / q:
            l = l_temp
            break
    else:
        raise "l not found"

    randnum = random.uniform(0, 1)
    if randnum < (1 - (elem * q - l)):
        return (l + 1) / q
    return l / q
