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


def Solve(c, A, b):
    if np.all(b >= 0):
        m = b.shape[0]
        n = c.shape[0]
        c = np.concatenate((c, np.zeros(m)))
        A = np.hstack((A, np.eye(m)))
        status, solution, objective = PrimalSimplex(c, A, b)
        if status == "optimal":
            solution = solution[:n]
        return status, solution, objective

    # Иначе запускаем фазу 1
    return Phase1(c, A, b)


def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Primal Simplex method.")
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

    print("Solving the linear program using the Primal Simplex method...\n")
    status, solution, objective = Solve(c, A, b)
    print("\nResult:")
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
