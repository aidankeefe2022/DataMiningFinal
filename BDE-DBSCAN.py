import random

import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl


def BDE_DBSCAN(X,y):
    n_pop = len(X.shape[0])
    max_iter = 10
    nVar = 2






X, y = skl.datasets.make_blobs(n_samples=random.randint(500,1000), random_state=200)
BDE_DBSCAN(X, y)