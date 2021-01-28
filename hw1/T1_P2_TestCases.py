import numpy as np
from T1_P2 import predict_knn, predict_kernel


'''
Instructions for using this Autograder:

1. Make sure it is in the same immediate directory/folder as your implementation file, which *must* be called T1_P2.py
2. Run this only after you have implemented the functions predict_kernel and predict_knn.

'''

def checker_t1_p2():
    alpha_05 = [0.378535057428291, 0.376197422450398, 0.42036257287011297, 0.35926744540567057, 0.422350293047517, 0.38954930008175787, 0.3523431433709818, 0.38384738253165507, 0.41413924288870996, 0.4036385707182667, 0.40725052310675924, 0.4195589072816149, 0.42161655410096394]

    alpha_6 = [0.16975380835973425, 0.17101298992814784, 0.7393447700965897, 0.14857971068654477, 0.3142944017836196, 0.19633929752733367, 0.099680465564935, 0.24265001978627673, 0.34539242945067994, 0.6107809806413983, 0.6918820955321691, 0.7316525782396872, 0.7567231193522315]

    knn_2 = [0.205, 0.185, 0.8049999999999999, 0.19, 0.27999999999999997, 0.19, 0.005, 0.075, 0.205, 0.8300000000000001, 0.755, 0.755, 0.8300000000000001]

    knn_8 = [0.21874999999999997, 0.23250000000000004, 0.4975, 0.22375, 0.33375, 0.23375, 0.18750000000000003, 0.2825, 0.315, 0.5037499999999999, 0.485, 0.49249999999999994, 0.45625000000000004]

    if len(alpha_05)!=len(predict_kernel(alpha=0.5)):
        print("Your predictions for alpha = 0.5 have the wrong shape")
    else:
        alpha05_checker = np.linalg.norm(np.array(alpha_05) - np.array(predict_kernel(alpha=0.5)), ord=np.inf) < 0.001
        if alpha05_checker:
            alpha05_checker = "Pass"
        else:
            alpha05_checker = "Fail"

    if len(alpha_6)!=len(predict_kernel(alpha=6)):
        print("Your predictions for alpha = 6 have the wrong shape")
    else:
        alpha6_checker = np.linalg.norm(np.array(alpha_6) - np.array(predict_kernel(alpha=6)), ord=np.inf) < 0.001
        if alpha6_checker:
            alpha6_checker = "Pass"
        else:
            alpha6_checker = "Fail"

    if len(knn_2)!=len(predict_knn(2)):
        print("Your predictions for k = 2 have the wrong shape")
    else:
        knn2_checker = np.linalg.norm(np.array(knn_2) - np.array(predict_knn(2)), ord=np.inf) < 0.001
        if knn2_checker:
            knn2_checker = "Pass"
        else:
            knn2_checker = "Fail"

    if len(knn_8)!=len(predict_knn(8)):
        print("Your predictions for k = 8 have the wrong shape")
    else:
        knn8_checker = np.linalg.norm(np.array(knn_8) - np.array(predict_knn(8)), ord=np.inf) < 0.001
        if knn8_checker:
            knn8_checker = "Pass"
        else:
            knn8_checker = "Fail"

    print("Your test case results are, for alpha = 0.5 and alpha = 6 respectively:", alpha05_checker, alpha6_checker)
    print("Your test case results for k = 2 and k = 8 respectively are:", knn2_checker, knn8_checker)

checker_t1_p2()