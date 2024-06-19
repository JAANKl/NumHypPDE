import numpy as np
import matplotlib.pyplot as plt

solve_with_hand_calculations = True
check_hand_calculations = True

U_L = np.array([1,
                0])
U_R = np.array([0, 
                1])
U_0 = np.array([U_L, U_R]).T
print(U_0)

A = np.array([[2, 0], 
              [0, 2]])

R_by_hand = np.array([[1, 0],
                      [0, 1]])
R_inv_by_hand = np.array([[1, 0],
                      [0, 1]])
Lambda_by_hand = np.diag([2, 2])

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
