import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def directSphere(d, r_i=0, r_o=1):
    """
    Implementation: Krauth, Werner. Statistical Mechanics: Algorithms and
    Computations. Oxford Master Series in Physics 13. Oxford: Oxford University
    Press, 2006. page 42
    """
    # vector of univariate gaussians:
    rand = np.random.normal(size=d)
    # get its euclidean distance:
    dist = np.linalg.norm(rand, ord=2)
    # divide by norm
    normed = rand / dist

    # sample the radius uniformly from 0 to 1
    rad = np.random.uniform(r_i, r_o**d) ** (1 / d)
    # the r**d part was not there in the original implementation.
    # I added it in order to be able to change the radius of the sphere
    # multiply with vect and return
    return normed * rad


def quadForm(M, x):
    """
    Helper function to compute quadratic forms such as x^TMx
    """
    return np.dot(x, np.dot(M, x))


def sampleFromEllipsoid(S, rho, rInner=0, rOuter=1):
    lamb, eigV = np.linalg.eigh(S / rho)
    d = len(S)
    xy = directSphere(d, r_i=rInner, r_o=rOuter)  # sample from outer shells
    # transform sphere to ellipsoid
    # (refer to e.g. boyd lectures on linear algebra)
    T = np.linalg.inv(np.dot(np.diag(np.sqrt(lamb)), eigV.T))
    return np.dot(T, xy.T).T
