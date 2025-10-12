import numpy as np
import argparse

eps = 1e-9


# ===================================================
#                ПРЯМОЙ СИМПЛЕКС-МЕТОД
# ===================================================
def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    m, n = A.shape

    if basis is None:
        basis = list(range(n, n + m))
    if nbasis is None:
        nbasis = list(range(0, n))

    while True:
        B = A[:, basis]
        x_B = np.linalg.solve(B, b)
        if np.any(x_B < -eps):
            return "infeasible", None, None

        N = A[:, nbasis]
        c_B = c[basis]
        c_N = c[nbasis]
        y = np.linalg.solve(B.T, c_B)
        reduced_costs = c_N - N.T @ y

        if np.all(reduced_costs <= eps):
            x = np.zeros(len(c))
            x[basis] = x_B
            return "optimal", x, c @ x

        j = np.argmax(reduced_costs)
        entering = nbasis[j]
        d = np.linalg.solve(B, A[:, entering])
        if np.all(d <= eps):
            return "unbounded", None, None

        ratios = [x_B[i] / d[i] if d[i] > eps else np.inf for i in range(m)]
        leaving_idx = np.argmin(ratios)
        leaving = basis[leaving_idx]

        basis[leaving_idx] = entering
        nbasis[j] = leaving


# ===================================================
#                        ФАЗА 1
# ===================================================
def Phase1(c, A, b):
    m, n = A.shape
    for i in range(m):
        if b[i] < 0:
            A[i, :] *= -1
            b[i] *= -1

    A_ext = np.hstack([A, np.eye(m)])
    c_ext = np.hstack([np.zeros(n), -np.ones(m)])
    basis = list(range(n, n + m))
    nbasis = list(range(0, n))
    status, x_ext, obj = PrimalSimplex(c_ext, A_ext, b, basis, nbasis)

    if status != "optimal" or abs(obj) > eps:
        return "infeasible", None, None

    x = x_ext[:n]
    A_no_a = A_ext[:, :n]
    c_real = c
    basis = list(range(n - m, n)) if n >= m else list(range(n, n + m))
    nbasis = [j for j in range(A_no_a.shape[1]) if j not in basis]

    return PrimalSimplex(c_real, A_no_a, b, basis, nbasis)


# ===================================================
#                  ОСНОВНОЕ РЕШЕНИЕ
# ===================================================
def Solve(c, A, b):
    if np.all(b >= 0):
        A = np.hstack([A, np.eye(len(b))])
        c = np.hstack([c, np.zeros(len(b))])
        return PrimalSimplex(c, A, b)
    else:
        return Phase1(c, A, b)


# ===================================================
#                 ОСНОВНОЙ ВЫЗОВ (CLI)
# ===================================================
def main():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Primal Simplex method.")
    parser.add_argument("filename", type=str, help="Input file containing the LP problem.")
    args = parser.parse_args()

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

    print("===== INPUT DATA =====")
    print(f"n = {n}")
    print(f"m = {m}")
    print(f"c = {c}")
    print("A =")
    print(A)
    print(f"b = {b}")
    print("======================\n")

    print("Solving the linear program using the Primal Simplex method...\n")
    status, solution, objective = Solve(c, A, b)

    print("\n===== RESULT =====")
    print("Status:", status)
    if status == "optimal":
        print("Optimal solution x* =", solution)
        print("Optimal value =", objective)
    elif status == "unbounded":
        print("The problem is unbounded.")
    else:
        print("No feasible solution.")
    print("===================")


if __name__ == "__main__":
    main()
