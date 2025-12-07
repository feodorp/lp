import numpy as np
import sys
import argparse

from numpy.linalg.linalg import solve

# global vars
eps = 0.00001


def update_reverse(B_inv, d, v):
    return B_inv - np.outer(d, v.T @ B_inv) / (1 + v.T @ d)


def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    #  max c^T x
    #  Ax = b
    #  x >= 0
    #  n - число переменных
    #  m - число ограничений
    #  Да, действительно, считаем что n > m (с учетом слаков)
    # Предполагаем что проблема точно feasible но возможно unbounded
    m, n = A.shape
    n -= m
    if basis is None or nbasis is None:
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))

    A_B_inv = np.linalg.inv(A[:, basis])

    while True:
        # Подсказка: np.linalg.solve(M, v) решает систему Mx = v

        reduced_cost = c[nbasis] - A[:, nbasis].T @ A_B_inv.T @ c[basis]
        if np.all(reduced_cost < eps):
            break

        entering_index = np.argmax(reduced_cost)

        d = A_B_inv @ A[:, nbasis[entering_index]]
        if np.all(d < eps):
            return "unbounded", [], 0

        mask = d > 0
        mask_leaving_index = np.argmin((A_B_inv @ b)[mask] / d[mask])
        leaving_index = np.where(mask)[0][mask_leaving_index]

        A_B_inv = update_reverse(A_B_inv, A_B_inv @ (A[:, nbasis[entering_index]] - A[:, basis[leaving_index]]),
                                 np.where(np.arange(m) == leaving_index, 1, 0))
        basis[leaving_index], nbasis[entering_index] = nbasis[entering_index], basis[leaving_index]

    x = np.zeros(n + m)
    x[basis] = A_B_inv @ b
    return "optimal", x, c.T @ x


def DualSimplex(c, A, b, basis=None, nbasis=None):
    #  max c^T x
    #  Ax = b
    #  x >= 0
    #  n - число переменных
    #  m - число ограничений
    #  Да, действительно, считаем что n > m (с учетом слаков)
    # Предполагаем что проблема точно feasible но возможно unbounded
    m, n = A.shape
    n -= m
    if basis is None or nbasis is None:
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))

    B_inv = np.linalg.inv(A[:, basis])

    x_B = B_inv @ b

    N = A[:, nbasis]
    c_B = c[basis]
    c_N = c[nbasis]
    z_N = (B_inv @ N).T @ c_B - c_N

    while True:
        # 1
        if np.all(x_B >= -eps):
            break

        # 2
        i_B = np.argmin(x_B)
        i = basis[i_B]

        # 3
        e_i = np.zeros(m)
        e_i[i_B] = 1.0
        delta_z_N = -(B_inv @ N).T @ e_i.reshape(-1, 1)

        # 4
        delta_z_N = delta_z_N.flatten()
        mask = delta_z_N > eps
        if not np.any(mask):
            return "unbounded", [], 0.0

        ratios_s = z_N[mask] / delta_z_N[mask]
        s = np.min(ratios_s)
        if s <= eps:
            return "unbounded", [], 0.0


        # 5
        j_pos_in_mask = np.argmin(ratios_s)
        j_N = np.arange(len(nbasis))[mask][j_pos_in_mask]
        j = nbasis[j_N]

        # 6
        e_j = np.zeros(len(nbasis))
        e_j[j_N] = 1.0
        delta_x_B = (B_inv @ N) @ e_j.reshape(-1, 1)

        # 7
        delta_x_i = delta_x_B[i_B]
        t = x_B[i_B] / delta_x_i

        # 8
        x_B = x_B - t * delta_x_B
        x_j = t
        z_N = z_N - s * delta_z_N
        z_i = s

        # 9
        B_inv = update_reverse(B_inv, B_inv @ (A[:, nbasis[j_N]] - A[:, basis[i_B]]),
                                 np.where(np.arange(m) == i_B, 1, 0))
        basis[i_B], nbasis[j_N] = nbasis[j_N], basis[i_B]

        x_B = B_inv @ b
        c_B = c[basis]
        c_N = c[nbasis]
        N = A[:, nbasis]
        z_N = (B_inv @ N).T @ c_B - c_N


    x = np.zeros(n + m)
    x[basis] = B_inv @ b
    return "optimal", x, c.T @ x


def Phase1(c, A, b):
    n = c.shape[0]
    m = b.shape[0]
    new_c = np.concatenate((np.zeros(n + m), np.array([-1])))
    new_A = np.hstack((A, np.eye(m), -np.ones((m, 1))))
    p = n + np.argmin(b)
    basis = list(range(n, p)) + list(range(p + 1, n + m + 1))
    nbasis = list(range(n)) + list(range(p, p + 1))

    status, x, obj = PrimalSimplex(new_c, new_A, b, basis, nbasis)
    if status != "optimal" or obj > eps:
        return "infeasible", None, None

    c = np.concatenate((c, np.zeros(m + 1)))
    A = np.hstack((A, np.eye(m), np.zeros((m, 1))))
    basis = np.where(x != 0)[0]
    nbasis = np.where(x == 0)[0]
    d = m - basis.shape[0]
    basis = np.concatenate((basis, nbasis[:d]))
    nbasis = nbasis[d:]
    status, solution, objective = PrimalSimplex(c, A, b, basis, nbasis)
    if status == "optimal":
        solution = solution[:n]
    return status, solution, objective



def Solve(c, A, b, mode="auto"):
    m, n = A.shape

    c_full = np.concatenate((c, np.zeros(m)))
    A_full = np.hstack((A, np.eye(m)))

    basis = list(range(n, n + m))
    nbasis = list(range(0, n))

    B_inv = np.linalg.inv(A_full[:, basis])
    x_B = B_inv @ b

    if mode == "Primal":
        if np.all(x_B >= -eps):
            status, x_full, obj = PrimalSimplex(c_full, A_full, b, basis, nbasis)
        else:
            return Phase1(c_full, A_full, b)

    elif mode == "Dual":
        if np.all(x_B >= -eps):
            status, x_full, obj = PrimalSimplex(c_full, A_full, b, basis, nbasis)
        else:
            status, x_full, obj = DualSimplex(c_full, A_full, b, basis, nbasis)

    if status == "optimal":
        return status, x_full[:n], obj
    return status, [], obj

def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Dual Simplex method.")
    parser.add_argument("filename", type=str, help="Input file containing the LP problem.")
    return parser.parse_args()


def main():
    # boilerplate for reading input data
    args = proc_cmd()
    with open(args.filename, 'r', encoding='utf-8') as f:
        n, m = map(int, f.readline().split())
        c = np.array(list(map(float, f.readline().split())))
        A = []
        b = []
        for _ in range(m):
            *row, bi = map(float, f.readline().split())
            A.append(row)
            b.append(bi)
        A = np.array(A)
        b = np.array(b)
    print("n =", n)
    print("m =", m)
    print("c =", c)
    print("A =", A)
    print("b =", b)

    print("\n\nSolving the linear program using the Dual Simplex method...\n")
    status, solution, objective = Solve(c, A, b, "Dual")
    print("Result:")
    print("Status:", status)
    if status == "optimal":
        print("Optimal solution x* =", solution)
        print("Optimal value =", objective)
    elif status == "unbounded":
        print("The problem is unbounded.")
    else:
        print("No solution found.")



if __name__ == '__main__':
    main()