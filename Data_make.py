import random
import numpy as np
import sklearn as skl
import sklearn.datasets as skds


def create_rand_blob(x):
    X1, y1 = skds.make_blobs(n_samples=random.randint(500,1000), random_state=200, centers=x)
    return np.vstack((X1))

def create_rand_diff_blob():
    centers = [(0, 0), (5, 5), (10, 0)]
    stds = [0.5, 1.5, 0.3]
    X1, y1 = skds.make_blobs(n_samples=random.randint(500,1000), random_state=200, centers=[centers[0]], cluster_std=[stds[0]])
    X2, y2 = skds.make_blobs(n_samples=random.randint(500,1000), random_state=200, centers=[centers[1]], cluster_std=[stds[1]])
    X3, y3 = skds.make_blobs(n_samples=random.randint(500,1000), random_state=200, centers=[centers[2]], cluster_std=[stds[2]])
    return np.vstack((X1, X2, X3))

def create_rand_moon():
    X, y = skds.make_moons(n_samples=random.randint(500,1000), random_state=200)
    return np.vstack((X)) * 5

def create_rand_diff_moon():
    # Generate the first moon with default settings
    X1, y1 = skds.make_moons(n_samples=random.randint(500,1000), noise=0.05)

    # Generate the second moon, more densely packed
    X2, y2 = skds.make_moons(n_samples=random.randint(500,1000), noise=0.05)
    X2 = X2 * 0.5 + np.array([1, 0.5])

    X3, y3 = skds.make_moons(n_samples=random.randint(500,1000), noise=0.1)
    X3 = X3 * .2 + np.array([1, 0.2])
    return np.vstack((X1, X2, X3)) * 5

def create_rand_circle():
    X, y = skds.make_circles(n_samples=random.randint(500,1000), random_state=200, noise=random.random()*.2)
    return np.vstack((X)) * 5




