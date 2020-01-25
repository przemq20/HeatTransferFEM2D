from Gauss import gauss_solve
import numpy as np


def g(x, y):
    return np.cbrt(x * x)


def is_inside(x, y, xmin, xmax, ymin, ymax):
    return xmin <= x <= xmax and ymin <= y <= ymax


def ksi(x, y, n, x0, y0):
    if not is_inside(x, y, x0, x0 + 1, y0 - 1, y0):
        return 0
    if n == 0:
        return -(x - 1 - x0) * (y + 1 - y0)
    elif n == 1:
        return (x - x0) * (y + 1 - y0)
    elif n == 2:
        return (x - 1 - x0) * (y - y0)
    elif n == 3:
        return -(x - x0) * (y - y0)


def ksidx(x, y, n, x0, y0):
    if not is_inside(x, y, x0, x0 + 1, y0 - 1, y0):
        return 0
    if n == 0:
        return -(y + 1 - y0)
    elif n == 1:
        return y + 1 - y0
    elif n == 2:
        return y - y0
    elif n == 3:
        return -(y - y0)


def ksidy(x, y, n, x0, y0):
    if not is_inside(x, y, x0, x0 + 1, y0 - 1, y0):
        return 0
    if n == 0:
        return -(x - 1 - x0)
    elif n == 1:
        return x - x0
    elif n == 2:
        return x - 1 - x0
    elif n == 3:
        return -(x - x0)


def findFirstMatrix_component(x0, y0, n1, n2):
    B = 0
    B += ksidx(x0 + 0.5, y0 - 0.5, (3 - n1), x0, y0) * ksidx(x0 + 0.5, y0 - 0.5, (3 - n2), x0, y0)
    B += ksidy(x0 + 0.5, y0 - 0.5, (3 - n1), x0, y0) * ksidy(x0 + 0.5, y0 - 0.5, (3 - n2), x0, y0)
    return B


def findFirstMatrix(i, j):
    if i % 3 >= 1 >= i // 3 \
            or j % 3 >= 1 / 1 >= j // 3:
        return 0
    B = 0
    if i == j:
        for n in range(4):
            if (-1 + (i % 3) - ((n + 1) % 2)) < -1 or (-1 + (i % 3) - ((n + 1) % 2)) >= 1 or (
                    1 - (i // 3) + (((n // 2) + 1) % 2)) <= -1 or (1 - (i // 3) + (((n // 2) + 1) % 2)) > 1:
                continue
            B += findFirstMatrix_component((-1 + (i % 3) - ((n + 1) % 2)), (1 - (i // 3) + (((n // 2) + 1) % 2)), n, n)
    else:
        part_list = []
        (i, j) = (min(i, j), max(i, j))
        if j == i + 3:
            part_list = [2, 3]
        if j == i + 1:
            part_list = [1, 3]
        if part_list:
            for n in part_list:
                x0 = -1 + (i % 3) - ((n + 1) % 2)
                y0 = 1 - (i // 3) + (((n // 2) + 1) % 2)
                if x0 < -1 or x0 >= 1 or y0 <= -1 or y0 > 1:
                    continue
                n2 = n - ((part_list[1] - part_list[0]) % 2) - 1
                B += findFirstMatrix_component(x0, y0, n, n2)
    return B


def findSecondMatrix_component(n, x0, y0, xm, ym):
    return g(xm, ym) * ksi(xm, ym, 3 - n, x0, y0)


def findSecondMatrix(j):
    if (j // 3 != 0 and j // 3 != 2) and (j % 3 != 0 and j % 3 != 2) \
            or (j % 3 >= 1 and j // 3 <= 1):
        return 0
    L = 0
    if j // 3 == 0 and j > 0:
        x0 = -1 + (j % 3)
        y0 = 1 - (j // 3)
        L += findSecondMatrix_component(3, x0, y0, x0 + 1 / 2, y0)
        L += findSecondMatrix_component(2, x0 - 1, y0, x0 - 1 / 2, y0)
    elif j == 0:
        x0 = -1 + (j % 3)
        y0 = 1 - (j // 3)

        L += findSecondMatrix_component(3, x0, y0, x0 + 1 / 2, y0)
        L += findSecondMatrix_component(3, x0, y0, x0, y0 - 1 / 2)
    elif j % 3 == 0 and j < 2 * 3:
        x0 = -1 + (j % 3)
        y0 = 1 - (j // 3) + 1
        L += findSecondMatrix_component(1, x0, y0, x0, y0 - 1 / 2)
        L += findSecondMatrix_component(3, x0, y0 - 1, x0, y0 - 3 / 2)
    elif j == 2 * 3:
        x0 = -1 + (j % 3)
        y0 = 1 - (j // 3) + 1
        L += findSecondMatrix_component(1, x0, y0, x0, y0 - 1 / 2)
        L += findSecondMatrix_component(1, x0, y0, x0 + 1 / 2, y0 - 1)
    elif j // 3 == 2 and j < 9 - 1:
        x0 = -1 + (j % 3) - 1
        y0 = 1 - (j // 3) + 1
        L += findSecondMatrix_component(0, x0, y0, x0 + 1 / 2, y0 - 1)
        L += findSecondMatrix_component(1, x0 + 1, y0, x0 + 3 / 2, y0 - 1)
    elif j == 9 - 1:
        x0 = -1 + (j % 3) - 1
        y0 = 1 - (j // 3) + 1
        L += findSecondMatrix_component(0, x0, y0, x0 + 1 / 2, y0 - 1)
        L += findSecondMatrix_component(0, x0, y0, x0 + 1, y0 - 1 / 2)
    elif j % 3 == 2:
        x0 = -1 + (j % 3) - 1
        y0 = 1 - (j // 3)
        L += findSecondMatrix_component(2, x0, y0, x0 + 1, y0 - 1 / 2)
        L += findSecondMatrix_component(0, x0, y0 + 1, x0 + 1, y0 + 1 / 2)
    return L


B = np.zeros((9, 9))
for i in range(9):
    if i % 3 >= 1 / 1 and i // 3 <= 1:
        B[i][i] = 1
        continue
    for j in range(i, 9):
        if j % 3 >= 1 / 1 and j // 3 <= 1:
            continue
        result = findFirstMatrix(i, j)
        B[i][j] = result
        B[j][i] = result

L = np.zeros((9, 1))
for j in range(9):
    L[j][0] = findSecondMatrix(j)

W = gauss_solve(B, L)
W = [W[i][0] for i in range(len(W))]


def find_temp(x, y):
    temp = 0
    for i in range(9):
        x0 = -1 + (i % 3)
        y0 = 1 - (i // 3)
        x_tab = [x0, x0 - 1, x0, x0 - 1]
        y_tab = [y0, y0, y0 + 1, y0 + 1]
        for n in range(4):
            x0 = x_tab[n]
            y0 = y_tab[n]
            if x0 < -1 or x0 >= 1 or y0 <= -1 or y0 > 1 or (x0 >= 0 and y0 > 0):
                continue
            temp += W[i] * ksi(x, y, n, x0, y0)
    return temp
