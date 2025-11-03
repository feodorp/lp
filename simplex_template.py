import numpy as np
import sys
import argparse

# global vars
eps = 10**-6

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


def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    #  max c^T x
    #  Ax = b
    #  x >= 0
    #  n - число переменных
    #  m - число ограничений
    #  Да, действительно, считаем что n > m (с учетом слаков)
    # Предполагаем что проблема точно feasible но возможно unbounded
    m, n_full = A.shape
    n = n_full - m
    if basis is None or nbasis is None:
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))

    B = A[:, basis]
    L, U = lu_decomposition(B)

    max_iter = 200000
    it = 0

    while True:
        it += 1
        if it > max_iter:
            return "stopped_by_iter_limit", None, None

        cB, cN = c[basis], c[nbasis]
        y = solve_transpose(L, U, cB)
        reduced = cN - A[:, nbasis].T @ y

        if np.max(reduced) < eps:
            bbar = solve(L, U, b)
            x_full = np.zeros(n + m)
            x_full[basis] = bbar
            x_dir = x_full[:n]
            return "optimal", x_dir, float(c[:n] @ x_dir)

        entering_index = min(j for j, rc in zip(nbasis, reduced) if rc > eps)
        entering_pos = nbasis.index(entering_index)
        a_j = A[:, entering_index]
        d = solve(L, U, a_j)
        if np.max(d) < eps:
            return "unbounded", None, None

        bbar = solve(L, U, b)
        ratios = [(bbar[i] / d[i], basis[i], i) for i in range(m) if d[i] > eps]
        if not ratios:
            return "unbounded", None, None
        theta, leaving_index, p = min(ratios, key=lambda t: (t[0], t[1]))

        basis_new = basis.copy()
        basis_new[p] = entering_index
        try:
            L_new, U_new = update(L, U, a_j, p)
            Bnew = A[:, basis_new]
            resid = np.linalg.norm(Bnew - L_new @ U_new, ord=np.inf)
            if resid > 1e-6:
                raise np.linalg.LinAlgError
            basis, L, U = basis_new, L_new, U_new
            nbasis[entering_pos] = leaving_index
        except np.linalg.LinAlgError:
            Bnew = A[:, basis_new]
            L_new, U_new = lu_decomposition(Bnew)
            basis, L, U = basis_new, L_new, U_new
            nbasis[entering_pos] = leaving_index


def Phase1(c, A, b):
    m, n = A.shape
    I = np.eye(m)

    A_slack = np.hstack([A, I])
    A_phase1 = np.hstack([A_slack, I])
    c_phase1 = np.concatenate([np.zeros(n + m), -np.ones(m)])

    basis = list(range(n + m, n + 2 * m))
    nbasis = list(range(0, n + m))

    status, _, obj = PrimalSimplex(c_phase1, A_phase1, b, basis, nbasis)

    if status != "optimal" or obj < -eps:
        return "infeasible", None, None

    A_new = A_slack
    c_new = np.concatenate([c, np.zeros(m)])

    basis_new = []
    for j in basis:
        if j < n + m:
            basis_new.append(j)
        else:
            for k in range(n + m):
                if k not in basis_new and abs(A_new[:, k][basis.index(j)]) > eps:
                    basis_new.append(k)
                    break

    nbasis_new = [j for j in range(n + m) if j not in basis_new]

    return PrimalSimplex(c_new, A_new, b, basis_new, nbasis_new)


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
