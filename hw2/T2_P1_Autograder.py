import numpy as np
from T2_P1 import LogisticRegressor, basis1, basis2, basis3

'''
Instructions for using this Autograder:

0. In this problem, this autograder is used as a sanity check for your implementation of gradient descent.
1. Make sure it is in the same immediate directory/folder as your implementation file, which *must* be called T2_P1.py
2. Run this only after you have implemented basis2, basis3, and the LogisticRegressor class.
'''

def checker_t2_p1():
    eta = 0.001
    runs = 10000
    
    # Input Data
    x = np.array([-8, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    y = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)
    
    TestWs = [np.array([[0.47662057],
       [0.10811676]]),np.array([[-0.35625859],
       [ 0.61210836],
       [ 0.20969945],
       [ 0.0074232 ]]),np.array([[-0.52491169],
       [ 0.69685124],
       [-1.53142516],
       [10.73682861],
       [ 5.16062475],
       [ 0.3440001 ]])]
    
    modelTest = LogisticRegressor(eta=eta,runs=runs)

    x1 = basis1(x)
    modelTest.fit(x1,y,w_init=np.ones((x1.shape[1], 1)))
    
    if len(TestWs[0])!=len(modelTest.W):
        print("Your w for basis1 has the wrong shape")
    else:
        basis1_checker = np.allclose(TestWs[0], modelTest.W, rtol=0, atol=1e-2)
        if basis1_checker:
            basis1_checker = "Pass"
        else:
            basis1_checker = "Fail"
    
    x2 = basis2(x)
    modelTest.fit(x2,y,w_init=np.ones((x2.shape[1], 1)))
    
    if len(TestWs[1])!=len(modelTest.W):
        print("Your w for basis2 has the wrong shape")
    else:
        basis2_checker = np.allclose(TestWs[1], modelTest.W, rtol=0, atol=1e-2)
        if basis2_checker:
            basis2_checker = "Pass"
        else:
            basis2_checker = "Fail"
    
    x3 = basis3(x)
    modelTest.fit(x3,y,w_init=np.ones((x3.shape[1], 1)))
    
    if len(TestWs[2])!=len(modelTest.W):
        print("Your w for basis3 has the wrong shape")
    else:
        basis3_checker = np.allclose(TestWs[2], modelTest.W, rtol=0, atol=1e-2)
        if basis3_checker:
            basis3_checker = "Pass"
        else:
            basis3_checker = "Fail"
    
    print("Your test case results are, for basis 1, 2, and 3 respectively:", basis1_checker, basis2_checker, basis3_checker)

checker_t2_p1()
