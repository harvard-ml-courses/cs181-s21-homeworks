import numpy as np
import math

data = [(0., 0., 0.),
        (0., 0.5, 0.),
        (0., 1., 0.),
        (0.5, 0., 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 1., 0.5),
        (1., 0., 1.),
        (1., 0.5, 1.),
        (1., 1., 1.)]

alpha = 10

W1 = alpha * np.array([[1., 0.], [0., 1.]])
W2 = alpha * np.array([[0.1, 0.], [0., 1.]])
W3 = alpha * np.array([[1., 0.], [0., 0.1]])

def weighted_dist(xi, xj, W):
    diff = xi - xj
    dist = np.dot(np.dot(diff.T, W), diff)
    return np.ndarray.item(dist)

def compute_loss(W):
    data2 = [((np.array([[x1, x2]])).T, y) for (x1, x2, y) in data]
    loss = 0
    for xi, yi in data2:
        sum1 = 0
        sum2 = 0
        for xj, yj in data2:
            if (xi == xj).all():
                continue
            kernel = math.exp(-1 * weighted_dist(xi, xj, W))
            sum1 += kernel * yj
            sum2 += kernel
        loss += (yi - sum1 / sum2) ** 2
    return loss

print(compute_loss(W1))
print(compute_loss(W2))
print(compute_loss(W3))
