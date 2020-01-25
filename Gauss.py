import numpy as np


def gauss_prepare_matrix(M):
    n = M.shape[0]
    if n == 1:
        return
    for i in range(n):
        if M[i][0] != 0:
            if i != 0:
                for j in range(n + 1):
                    M[i][j], M[j][i] = M[j][i], M[i][j]
            break
    for i in range(1, n):
        mult = M[i][0] / M[0][0]
        for j in range(0, n + 1):
            M[i][j] = M[i][j] - mult * M[0][j]
    gauss_prepare_matrix(M[1:, 1:])


def gauss_solve(B, L):
    M = np.append(B, L, axis=1)
    gauss_prepare_matrix(M)
    n = M.shape[0]
    W = np.zeros((n, 1))
    for i in range(n - 1, -1, -1):
        W[i][0] = M[i][n]
        for j in range(i + 1, n):
            W[i][0] -= M[i][j] * W[j][0]
        W[i][0] /= M[i][i]
    return W
