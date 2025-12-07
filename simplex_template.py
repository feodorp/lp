import numpy as np
import argparse

eps = 0.00001


def augment_with_slacks(c, A, b):
    b = np.asarray(b, dtype=float).reshape(-1)
    c = np.asarray(c, dtype=float).reshape(-1)

    m, n = A.shape

    I = np.eye(m, dtype=float)
    A_eq = np.hstack([A, I])
    c_aug = np.concatenate([c, np.zeros(m, dtype=float)])

    basis = list(range(n, n + m))
    nbasis = list(range(0, n))

    return c_aug, A_eq, b, basis, nbasis


def DualSimplex(c, A, b, U_value=1e8, max_iter=1000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    m, n = A.shape
    n_full = n + m

    # 5.1

    c_slack, A_eq, b, basis, nbasis = augment_with_slacks(c, A, b)
    U_full = np.full(n_full, U_value, dtype=float)

    # 5.2

    d = np.zeros(n_full, dtype=float)

    # B = A_eq[:, basis]
    N = A_eq[:, nbasis]
    c_N = c_slack[nbasis]

    # y = B^-T c_B, но c_B = 0, тогда y = 0
    y = np.zeros(m)

    bar_c_N = c_N - N.T @ y

    x_N = np.zeros(len(nbasis))
    r = np.zeros(len(nbasis))
    for pos, j in enumerate(nbasis):
        if bar_c_N[pos] <= eps:
            d[j] = 1.0
            x_N[pos] = 0.0
        else:
            d[j] = -1.0
            x_N[pos] = U_full[j]
        r[pos] = d[j] * bar_c_N[pos]

    # поскольку B = I
    # x_B = b - N @ x_N

    # 5.3
    for it in range(max_iter):
        B = A_eq[:, basis]
        N = A_eq[:, nbasis]
        c_B = c_slack[basis]

        try:
            y = np.linalg.solve(B.T, c_B)
        except np.linalg.LinAlgError:
            return "infeasible", None, None

        bar_c_N = c_slack[nbasis] - N.T @ y
        r = np.empty(len(nbasis))
        x_N = np.empty(len(nbasis))
        for pos, j in enumerate(nbasis):
            r[pos] = d[j] * bar_c_N[pos]
            x_N[pos] = 0.0 if d[j] > 0 else U_full[j]

        try:
            x_B = np.linalg.solve(B, b - N @ x_N)
        except np.linalg.LinAlgError:
            return "infeasible", None, None

        if np.all(x_B >= -eps) and np.all(r <= eps):
            x_full = np.zeros(n_full)
            for row, j in enumerate(basis):
                x_full[j] = x_B[row]
            for pos, j in enumerate(nbasis):
                x_full[j] = x_N[pos]

            x_orig = x_full[:n]
            obj = float(c @ x_orig)

            if np.any(x_orig >= U_value - 1e-6 * U_value):
                return "unbounded", None, None

            return "optimal", x_orig, obj

        v_rows = [row for row in range(m) if x_B[row] < -eps]

        leave_row = min(v_rows, key=lambda k: basis[k])
        leaving = basis[leave_row]

        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            return "infeasible", None, None

        row_inv = B_inv[leave_row, :]
        hat_row = row_inv @ A_eq

        sigma = np.empty(len(nbasis))
        for pos, j in enumerate(nbasis):
            sigma[pos] = -hat_row[j] * d[j]

        candidates = []
        t_vals = []
        for pos in range(len(nbasis)):
            if sigma[pos] > eps:
                t_j = -r[pos] / sigma[pos]
                candidates.append(pos)
                t_vals.append(t_j)

        if not candidates:
            return "infeasible", None, None

        t_min = min(t_vals)
        best_pos = [
            pos for pos, t_j in zip(candidates, t_vals)
            if t_j <= t_min + eps]

        pos_star = min(best_pos, key=lambda k: nbasis[k])
        entering = nbasis[pos_star]

        basis[leave_row] = entering
        nbasis[pos_star] = leaving

        d[leaving] = +1.0

    return "iterations_limit", None, None


def Solve(c, A, b):
    return DualSimplex(c, A, b)


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
