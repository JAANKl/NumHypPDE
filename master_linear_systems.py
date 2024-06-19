import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
solve_with_hand_calculations = True
check_hand_calculations = True

c = 2

U_L = np.array([1,
                2])
U_R = np.array([3, 
                4])
U_0 = np.array([U_L, U_R]).T
print(U_0)

A = np.array([[0, c], 
              [c, 0]])

R_by_hand = np.array([[1, 1],
                      [-1, 1]])
R_inv_by_hand = np.array([[1/2, -1/2],
                          [1/2, 1/2]])
Lambda_by_hand = np.diag([-c, c])

if check_hand_calculations:

    #assert that eigenvalues are increasing
    assert Lambda_by_hand[0, 0] <= Lambda_by_hand[1, 1], "Eigenvalues are not increasing"
    #assert R has orthogonal columns, but not necessarily normalized, only true if symmetric
    if np.allclose(A, A.T):
        assert np.allclose((R_by_hand.T @ R_by_hand)[0, 1]**2 + (R_by_hand.T @ R_by_hand)[1, 0]**2, 0), "R does not have orthogonal columns"
    assert np.allclose(np.linalg.det(A), np.linalg.det(Lambda_by_hand)), "Wrong eigenvalues"

    if np.allclose(R_by_hand @ R_inv_by_hand, np.eye(2)):
        print("Hand calculated R, R_inv are inverses")
    else:
        print("INCORRECT: Hand calculated R, R_inv are not inverses")

    if np.allclose(R_by_hand @ Lambda_by_hand @ R_inv_by_hand, A):
        print("Hand calculated R, Lambda and R_inv are correct")
    else:
        print("INCORRECT: Hand calculated R, Lambda and R_inv are not correct")




#eigenvalues
lambdas, R = np.linalg.eig(A)

#sorting from lowest to highest eigenvalue
idx = lambdas.argsort()
lambdas = lambdas[idx]
R = R[:,idx]

R_inv = np.linalg.inv(R)

if solve_with_hand_calculations:
    print("Solving with hand calculations")
    R = R_by_hand
    R_inv = R_inv_by_hand
    lambdas = Lambda_by_hand.diagonal()

W_L = R_inv @ U_L
W_R = R_inv @ U_R
W_0 = R_inv @ U_0

print(f"Eigenvalues: {lambdas}")
print(f"R: {R}")
print(f"R_inv: {R_inv}")

print(f"W_L: {W_L}")
print(f"W_R: {W_R}")
print(f"W_0: {W_0}")

U_star = R@ np.array([[1, 0],
                      [0, 0]]) @ R_inv @ U_R + R @ np.array([[0, 0],
                                                             [0, 1]]) @ R_inv @ U_L

print(f"U_star: {U_star}")

#explicit godunov flux with sympy
from sympy import init_printing,latex,symbols, pprint
init_printing(use_latex='mathjax')

U_j_1, U_j_2, U_j_p_1, U_j_p_2, U_j_m_1, U_j_m_2  = sp.symbols('U_{j}^1 U_{j}^2 U_{j+1}^1 U_{j+1}^2 U_{j-1}^2 U_{j-1}^2')
U_L = sp.Matrix([[U_j_1],
                    [U_j_2]])
U_R = sp.Matrix([[U_j_p_1],
                    [U_j_p_2]])

F_j_ph = 1/2*A@(U_L + U_R) - 1/2*R@np.diag(np.abs(lambdas))@R_inv*(U_R - U_L)
print(f"Godunov flux: F_jph = {F_j_ph}")
pprint(F_j_ph)
#update rule
dt, dx = sp.symbols('dt dx')
