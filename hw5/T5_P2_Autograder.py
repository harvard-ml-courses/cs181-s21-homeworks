#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 08:03:27 2021

@author: yashnair, rylan
"""

import numpy as np
import pickle
import sklearn.decomposition

from T5_P2 import get_cumul_var


def check_student_solution(num_leading_components=500):
    # set seed to control for randomness
    np.random.seed(42)

    # load data, reshape and call student's function
    mnist_pics = np.load("data/images.npy")
    num_images, height, width = mnist_pics.shape
    mnist_pics = np.reshape(mnist_pics, newshape=(num_images, height * width))

    # load expected cumulative variances
    # student cannot use this function!
    pca = sklearn.decomposition.PCA(n_components=num_leading_components)

    X = np.zeros(mnist_pics.shape)
    mu = np.mean(mnist_pics,axis=0)
    for i in range(len(mnist_pics)):
        X[i] = mnist_pics[i] - mu

    S = np.cov(np.matrix.transpose(X))
    total_variance = np.trace(S)

    pca.fit(mnist_pics)
    expected_cum_var = np.cumsum(pca.explained_variance_ / total_variance)

    # fetch student cumulative variance
    student_cum_var = get_cumul_var(
        mnist_pics=mnist_pics,
        num_leading_components=num_leading_components)

    # if student cumulative variance isn't a list or a numpy array, raise TypeError
    if not isinstance(student_cum_var, list) and not isinstance(student_cum_var, np.ndarray):
        raise TypeError(f'Student cumulative variance type ({type(student_cum_var)}) is invalid')

    # if student cumulative variance is a list, convert to numpy array
    if isinstance(student_cum_var, list):
        student_cum_var = np.array(student_cum_var)

    # check that student cumulative variance has correct number of leading components
    if student_cum_var.shape != (500,):
        raise ValueError(f'Student cumulative variance has wrong shape. Correct shape is (500,)')

    # assert that student cumulative variance matches expected cumulative variance
    assert np.allclose(student_cum_var, expected_cum_var, atol=1e-4)

    print("All test cases passed")


if __name__ == '__main__':
    check_student_solution()
