import numpy as np
import sys
import argparse

# global vars
eps = 0.00001
U = 1e20


def swap(b, n, j, i):
    b[j], n[i] = n[i], b[j]

def DualSimplex(c, A, b, basis=None, nbasis=None, return_basis=False):
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
    else:
        assert len(basis) == m, f"Basis has {len(basis)} variables instead of {m}."
        assert len(nbasis) == n, f"Non-Basis has {len(nbasis)} variables instead of {n}."

    idx_b, idx_n = np.array(basis), np.array(nbasis)

    B, N = A[:, idx_b], A[:, idx_n]
    c_b, c_n = c[idx_b], c[idx_n]

    B_inv = np.linalg.inv(B)

    while True:
        # z_b = np.zeros(n)
        z_n = (B_inv @ N).T @ c_b - c_n

        r_n = -z_n

        # initialization for dual feasibility
        x_n = np.zeros(n)
        x_n[r_n > eps] = U
        d = np.ones(n)
        d[x_n == U] = -1.0  # r_n > 0 => x_n = U => d = -1
        D = np.diagflat(d)

        x_b = B_inv @ (b - N @ x_n)
        r = D @ r_n
        assert not np.any(r > eps)

        # Находим кандидата для входа в dual базис
        i = np.argmin(x_b)
        if not x_b[i] < -eps:
            break

        # Вычисляем направление, не забывая детектировать unbounded
        a   =   B_inv @ N
        d_r = - a[i, :].ravel() * d  # sigma
        if not np.any(d_r > eps):    # dual is unbounded => primal is infeasible
            return "infeasible", None, None

        # Найти кандидата для выхода из базиса
        d_r_pos_args = np.array([j for j in range(len(d_r)) if d_r[j] > eps])
        d_r_pos, r_pos = d_r[d_r_pos_args], r[d_r_pos_args]
        theta = -r_pos / d_r_pos
        j = d_r_pos_args[np.argmin(theta)]
        assert not theta.min() < -eps, print(f"{theta=}")

        # d_x_b = - a[:, j].ravel() * d[j]

        # step sizes
        # t = theta.min()

        # updates
        # x_b += t * d_x_b
        # r   += t * d_r

        swap(idx_b, idx_n, i, j)

        B, N = A[:, idx_b], A[:, idx_n]
        c_b, c_n = c[idx_b], c[idx_n]

        # Sherman-Morrison
        a_i, a_j = B[:, i].reshape(-1, 1), N[:, j].reshape(-1, 1)
        u = a_i - a_j
        v = np.zeros((m, 1))
        v[i, 0] = 1

        B_inv = B_inv - (B_inv @ u) @ (v.T @ B_inv) / (1 + v.T @ B_inv @ u)

    # Восстановить исходную систему, восстановить x и вернуть результат
    z_star = c_b.T @ x_b
    x_star = np.zeros(n + m)
    x_star[idx_b] = x_b
    x_star[idx_n] = x_n

    x_star_initial = x_star[:n]

    # if any of components is equal to upper bound (which is considered infinity),
    # then the problem dual problem is infeasible => primal is unbounded
    if np.any(x_star_initial == U):
        return "unbounded", None, None

    result = "optimal", x_star_initial, z_star
    if return_basis:
        return *result, basis, nbasis
    return result

def Solve(c, A, b):
    # initial variables + slacks
    m = A.shape[0]
    c = np.concatenate([c, np.zeros(m)])
    A = np.concatenate([A, np.eye(m)], axis=1)
    return DualSimplex(c, A, b)

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

    print("Solving the linear program using the Dual Simplex method...\n")
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