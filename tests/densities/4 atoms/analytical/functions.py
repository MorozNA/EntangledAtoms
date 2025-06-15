import numpy as np


def pg(coeff, alpha, R, r1, r2):
    distance = np.array([r1 - R[0], r2 - R[1]])
    distance = distance ** 2
    prod = alpha[0] * distance[0] + alpha[1] * distance[1]
    return coeff * np.exp(-prod)

