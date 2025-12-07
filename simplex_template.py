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
    m, n = A.shape

    A_full = np.hstack([A, np.eye(m, dtype=float)])
    c_full = np.concatenate([c, np.zeros(m, dtype=float)])

    if m > 0:
        bigU = 1.0 + max(1.0, float(np.max(np.abs(b))))
    else:
        bigU = 2.0
    U_full = np.full(n + m, bigU, dtype=float)

    B = list(range(n, n + m))
    N = list(range(n))

    d = np.zeros(n + m, dtype=float)

    for it in range(max_iters):
        B_mat = A_full[:, B]
        try:
            L, Ufac = lu_decomposition(B_mat)
        except np.linalg.LinAlgError:
            return "infeasible", None, None

        y = solve_transpose(L, Ufac, c_full[B])

        N_mat = A_full[:, N]
        cbar_N = c_full[N] - N_mat.T @ y

        x_full = np.zeros(n + m, dtype=float)

        for k_pos, j in enumerate(N):
            if d[j] == 0.0:
                if cbar_N[k_pos] <= eps:
                    d[j] = +1.0
                    x_full[j] = 0.0
                else:
                    d[j] = -1.0
                    x_full[j] = U_full[j]
            else:
                x_full[j] = 0.0 if d[j] > 0 else U_full[j]

        rhs = b - A_full[:, N] @ x_full[N]
        try:
            x_B = solve(L, Ufac, rhs)
        except np.linalg.LinAlgError:
            return "infeasible", None, None

        for pos, j in enumerate(B):
            x_full[j] = x_B[pos]

        y = solve_transpose(L, Ufac, c_full[B])
        N_mat = A_full[:, N]
        cbar_N = c_full[N] - N_mat.T @ y
        r_N = np.array([d[N[i]] * cbar_N[i] for i in range(len(N))], dtype=float)

        if np.all(x_B >= -eps) and np.all(r_N <= eps):
            if np.all(cbar_N <= eps):
                x_star = x_full[:n]
                obj = float(np.dot(c, x_star))
                return "optimal", x_star, obj
            else:
                return "unbounded", None, None

        neg_rows = [i for i, val in enumerate(x_B) if val < -1e-8]
        if not neg_rows:
            return "infeasible", None, None

        i_pos = min(neg_rows, key=lambda idx: x_B[idx])
        i_glob = B[i_pos]

        e = np.zeros(m, dtype=float)
        e[i_pos] = 1.0
        w = solve_transpose(L, Ufac, e)

        sigma = []
        for j in N:
            a_j = A_full[:, j]
            a_hat_ij = float(np.dot(w, a_j))
            sigma.append(-a_hat_ij * d[j])
        sigma = np.array(sigma, dtype=float)

        cand = [k for k, sig in enumerate(sigma) if sig > 1e-8]
        if not cand:
            return "unbounded", None, None

        ratios = [(-r_N[k] / sigma[k], k) for k in cand]
        t, j_pos = min(ratios, key=lambda pair: (pair[0], N[pair[1]]))
        j_glob = N[j_pos]

        B[i_pos] = j_glob
        N[j_pos] = i_glob

        d[i_glob] = +1.0
        d[j_glob] = 0.0

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