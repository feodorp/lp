import numpy as np
import sys
import argparse

# global vars
eps = 1e-6


def lu_decomposition(B):
    m = B.shape[0]
    L = np.eye(m, dtype=float)
    U = np.zeros_like(B, dtype=float)

    for i in range(m):
        for j in range(i, m):
            s = float(np.dot(L[i, :i], U[:i, j]))
            U[i, j] = B[i, j] - s
        if abs(U[i, i]) < eps:
            raise np.linalg.LinAlgError("Zero pivot in lu_decomposition")
        for j in range(i + 1, m):
            s = float(np.dot(L[j, :i], U[:i, i]))
            L[j, i] = (B[j, i] - s) / U[i, i]

    return L, U


def forward_sub(L, b):
    m = L.shape[0]
    x = np.zeros_like(b, dtype=float)
    for i in range(m):
        s = float(np.dot(L[i, :i], x[:i]))
        if abs(L[i, i]) < eps:
            raise np.linalg.LinAlgError("Singular L in forward_sub")
        x[i] = (b[i] - s) / L[i, i]
    return x


def backward_sub(U, b):
    m = U.shape[0]
    x = np.zeros_like(b, dtype=float)
    for i in range(m - 1, -1, -1):
        s = float(np.dot(U[i, i + 1 :], x[i + 1 :]))
        if abs(U[i, i]) < eps:
            raise np.linalg.LinAlgError("Singular U in backward_sub")
        x[i] = (b[i] - s) / U[i, i]
    return x


def solve(L, U, rhs):
    v = forward_sub(L, rhs)
    return backward_sub(U, v)


def solve_transpose(L, U, rhs):
    m = U.shape[0]
    w = np.zeros_like(rhs, dtype=float)
    for i in range(m):
        s = float(np.dot(U[:i, i], w[:i]))
        if abs(U[i, i]) < eps:
            raise np.linalg.LinAlgError("Singular U^T in solve_transpose")
        w[i] = (rhs[i] - s) / U[i, i]
    y = np.zeros_like(rhs, dtype=float)
    for i in range(m - 1, -1, -1):
        s = float(np.dot(L[i + 1 :, i], y[i + 1 :]))
        y[i] = w[i] - s
    return y


def update(L, U, a_j, p):
    m = L.shape[0]
    v = forward_sub(L, a_j)
    H = U.copy()
    H[:, p] = v
    L_new = L.copy()
    for k in range(p, m - 1):
        pivot = H[k, k]
        if abs(pivot) < eps:
            raise np.linalg.LinAlgError("Zero pivot in update")
        alpha = H[k + 1, k] / pivot
        H[k + 1, k:] -= alpha * H[k, k:]
        L_new[:, k + 1] += alpha * L_new[:, k]
    return L_new, H


def DualSimplex(c, A, b, max_iters=5000):
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape

    A_full = np.hstack([A, np.eye(m, dtype=float)])
    c_full = np.concatenate([c, np.zeros(m, dtype=float)])
    total = n + m

    scale = 1.0
    if A.size > 0:
        scale = max(scale, float(np.max(np.abs(A))))
    if b.size > 0:
        scale = max(scale, float(np.max(np.abs(b))))
    if c.size > 0:
        scale = max(scale, float(np.max(np.abs(c))))
    bigU = 1e6 * scale
    U_full = np.full(total, bigU, dtype=float)

    basis = list(range(n, total))
    nbasis = list(range(n))

    for _ in range(max_iters):
        B_mat = A_full[:, basis]

        try:
            L, Ufac = lu_decomposition(B_mat)

            def bs(rhs):
                return solve(L, Ufac, rhs)

            def bsT(rhs):
                return solve_transpose(L, Ufac, rhs)

        except np.linalg.LinAlgError:

            def bs(rhs, B_mat=B_mat):
                return np.linalg.solve(B_mat, rhs)

            def bsT(rhs, B_mat=B_mat):
                return np.linalg.solve(B_mat.T, rhs)

        y = bsT(c_full[basis])

        N_mat = A_full[:, nbasis]
        c_bar = c_full[nbasis] - N_mat.T @ y

        d_N = np.where(c_bar > eps, -1.0, 1.0)
        x_N = np.where(c_bar > eps, U_full[nbasis], 0.0)

        r = d_N * c_bar

        rhs = b - N_mat @ x_N
        try:
            x_B = bs(rhs)
        except np.linalg.LinAlgError:
            return "infeasible", None, None

        if np.all(x_B >= -eps) and np.all(r <= eps):
            x_full = np.zeros(total, dtype=float)
            x_full[basis] = x_B
            x_full[nbasis] = x_N

            if np.any(x_full[:n] >= U_full[:n] - eps):
                return "unbounded", None, None

            x_star = x_full[:n]
            obj = float(np.dot(c, x_star))
            return "optimal", x_star, obj

        neg_rows = np.flatnonzero(x_B < -eps)
        if neg_rows.size == 0:
            return "infeasible", None, None
        leave_pos = int(neg_rows[0])
        leaving_idx = basis[leave_pos]

        e_i = np.zeros(m, dtype=float)
        e_i[leave_pos] = 1.0
        w = bsT(e_i)
        sigma = -(w @ N_mat) * d_N

        valid_mask = sigma > eps
        if not np.any(valid_mask):
            return "infeasible", None, None

        ratios = np.full(len(nbasis), np.inf)
        ratios[valid_mask] = -r[valid_mask] / sigma[valid_mask]

        min_ratio = np.min(ratios[valid_mask])
        cand_pos = np.flatnonzero((ratios <= min_ratio + 1e-12) & valid_mask)
        enter_pos = int(min(cand_pos, key=lambda pos: nbasis[pos]))
        entering_idx = nbasis[enter_pos]

        basis[leave_pos] = entering_idx
        nbasis[enter_pos] = leaving_idx

    return "max_iters", None, None


def Solve(c, A, b):
    return DualSimplex(c, A, b)


def proc_cmd():
    parser = argparse.ArgumentParser(
        description="Solve a linear program using the Primal Simplex method."
    )
    parser.add_argument(
        "filename", type=str, help="Input file containing the LP problem."
    )
    return parser.parse_args()


def main():
    # boilerplate for reading input data
    args = proc_cmd()
    with open(args.filename, "r", encoding="utf-8") as f:
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


if __name__ == "__main__":
    main()
