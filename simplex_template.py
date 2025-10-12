import numpy as np
import sys
import argparse

# global vars
eps = 10**-9

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
        s = float(np.dot(L[i, i + 1 :], y[i + 1 :]))
        y[i] = w[i] - s
    return y


def update(L, U, a_j, p):
    m = L.shape[0]
    v = forward_sub(L, a_j)
    H = U.copy()
    H[:, p] = v
    L_new = L.copy()
    for i in range(p, m - 1):
        pivot = H[i, p]
        if abs(pivot) < eps:
            raise np.linalg.LinAlgError("Zero pivot in update")
        below = H[i + 1, p]
        alpha = below / pivot
        H[i + 1, :] -= alpha * H[i, :]
        L_new[:, i + 1] += alpha * L_new[:, i]
    U_new = H
    return L_new, U_new


def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    #  max c^T x
    #  Ax = b
    #  x >= 0
    #  n - число переменных
    #  m - число ограничений
    #  Да, действительно, считаем что n > m (с учетом слаков)
    # Предполагаем что проблема точно feasible но возможно unbounded
    m, n = A.shape
    n = n - m

    if basis is None or nbasis is None:
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))

    B = A[:, basis]
    try:
        L, U = lu_decomposition(B)
    except np.linalg.LinAlgError:
        L, U = lu_decomposition(B + eps * np.eye(m))

    while True:
        cB = c[basis]
        cN = c[nbasis]

        try:
            y = solve_transpose(L, U, cB)
        except np.linalg.LinAlgError:
            y = solve_transpose(L + eps * np.eye(m), U + eps * np.eye(m), cB)

        reduced_cost = cN - A[:, nbasis].T @ y

        if np.max(reduced_cost) < eps:
            try:
                bbar = solve(L, U, b)
            except np.linalg.LinAlgError:
                bbar = solve(L + eps * np.eye(m), U + eps * np.eye(m), b)

            x_full = np.zeros(n + m)
            x_full[basis] = bbar
            x_direct = x_full[:n]
            obj = float(c[:n].dot(x_direct))
            return "optimal", x_direct, obj

        entering_pos = None
        for pos, val in enumerate(reduced_cost):
            if val > eps:
                entering_pos = pos
                break

        entering_index = nbasis[entering_pos]
        a_j = A[:, entering_index]

        try:
            d = solve(L, U, a_j)
        except np.linalg.LinAlgError:
            d = solve(L + eps * np.eye(m), U + eps * np.eye(m), a_j)

        if np.max(d) < eps:
            return "unbounded", None, None

        try:
            bbar = solve(L, U, b)
        except np.linalg.LinAlgError:
            bbar = solve(L + eps * np.eye(m), U + eps * np.eye(m), b)

        p = None
        theta = None
        for i in range(m):
            if d[i] > eps:
                val = bbar[i] / d[i]
                if (
                    (theta is None)
                    or (val < theta - eps)
                    or (abs(val - theta) <= eps and (p is None or i < p))
                ):
                    theta = val
                    p = i

        leaving_index = basis[p]

        try:
            L_new, U_new = update(L, U, a_j, p)
            basis[p] = entering_index
            nbasis[entering_pos] = leaving_index
            L, U = L_new, U_new
        except np.linalg.LinAlgError:
            basis_new = basis.copy()
            basis_new[p] = entering_index
            Bnew = A[:, basis_new]
            try:
                L_new, U_new = lu_decomposition(Bnew)
            except np.linalg.LinAlgError:
                L_new, U_new = lu_decomposition(Bnew + eps * np.eye(m))
            basis = basis_new
            nbasis[entering_pos] = leaving_index
            L, U = L_new, U_new


def Phase1(c, A, b):
    m, n = A.shape
    I = np.eye(m)
    A_slack = np.hstack([A, I])
    a0 = -np.ones((m, 1))
    new_A = np.hstack([A_slack, a0])
    new_c = np.concatenate([np.zeros(n + m), np.array([-1.0])])
    basis = list(range(n, n + m))
    nbasis = list(range(0, n)) + [n + m]

    if np.max(b) < eps:
        p = int(np.argmin(b))
        basis[p] = n + m
        nbasis = list(range(0, n)) + [n + i for i in range(m) if i != p]

    status, x, obj = PrimalSimplex(new_c, new_A, b, basis, nbasis)
    if status != "optimal" or obj < -eps:
        return "infeasible", None, None

    A = new_A[:, : (n + m)]
    c = np.concatenate([c, np.zeros(m)])
    basis = list(range(n, n + m))
    nbasis = list(range(0, n))
    return PrimalSimplex(c, A, b, basis, nbasis)


def Solve(c, A, b):
    if np.all(b >= -eps):
        m, n = A.shape
        A = np.hstack([A, np.eye(m)])
        c = np.concatenate([c, np.zeros(m)])
        return PrimalSimplex(c, A, b)
    
    # Иначе запускаем фазу 1
    return Phase1(c, A, b)


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
