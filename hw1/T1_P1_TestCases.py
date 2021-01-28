
import numpy as np
from T1_P1 import compute_loss

'''
Instructions for using this Autograder:

1. Make sure it is in the same immediate directory/folder as your implementation file, which *must* be called T1_P1.py
2. Run this only after you have implemented the function compute_loss, which returns the loss for a specified kernel.
3. The test cases this Autograder uses are distinct from the kernels specified in the homework.

'''

data = [(0., 0., 0.),
        (0., 0.5, 0.),
        (0., 1., 0.),
        (0.5, 0., 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 1., 0.5),
        (1., 0., 1.),
        (1., 0.5, 1.),
        (1., 1., 1.)]


W_test1 = 3*np.array([[1., 0.], [0., 1.]])
W_test2 = 3*np.array([[0.1, 0.], [0., 1.]])
W_test3 = 3*np.array([[1., 0.], [0., 0.1]])

testloss1 = 0.5979878781744292
testloss2 = 2.059976079470737
testloss3 = 0.36754440076084405

case1_checker = np.abs(testloss1 - compute_loss(W_test1)) < 0.001
if case1_checker:
    case1_checker = "Pass"
else:
    case1_checker = "Fail"
case2_checker = np.abs(testloss2 - compute_loss(W_test2)) < 0.001
if case2_checker:
    case2_checker = "Pass"
else:
    case2_checker = "Fail"
case3_checker = np.abs(testloss3 - compute_loss(W_test3)) < 0.01
if case3_checker:
    case3_checker = "Pass"
else:
    case3_checker = "Fail"

print("Your test case results are : ", case1_checker, case2_checker, case3_checker)
