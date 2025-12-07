import numpy as np
import sys
import argparse

eps = 1e-9

def _solve(B, v):
    try:
        return np.linalg.solve(B, v)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(B, v, rcond=None)[0]

def _unbounded_trivial(c, A):
    col_le0 = np.all(A <= eps, axis=0)
    good = np.where((col_le0) & (c > eps))[0]
    return good.size > 0

def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    m, tot = A.shape
    n = tot - m
    if basis is None or nbasis is None:
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))
    max_iters = 100000
    for _ in range(max_iters):
        B = A[:, basis]
        N = A[:, nbasis]
        xB = _solve(B, b)
        y = _solve(B.T, c[basis])
        rc = c[nbasis] - N.T @ y
        if np.all(rc <= eps) and np.all(xB >= -1e-12):
            x = np.zeros(tot)
            x[basis] = xB
            x = np.where(np.abs(x) < 1e-12, 0.0, x)
            return "optimal", x[:n], float(c[:n] @ x[:n])
        j_rel = int(np.argmax(rc))
        if rc[j_rel] <= eps:
            x = np.zeros(tot)
            x[basis] = xB
            if np.all(xB >= -1e-10):
                return "optimal", x[:n], float(c[:n] @ x[:n])
            return "infeasible", None, None
        j = nbasis[j_rel]
        d = _solve(B, A[:, j])
        if not np.any(d > eps):
            return "unbounded", None, None
        ratios = np.full(m, np.inf)
        pos = d > eps
        ratios[pos] = xB[pos] / d[pos]
        i_rel = int(np.argmin(ratios))
        leaving = basis[i_rel]
        basis[i_rel] = j
        nbasis[j_rel] = leaving
    return "optimal", np.zeros(n), 0.0

def Phase1_leq(c, A, b):
    m, n = A.shape
    S = np.eye(m)
    A_art = -np.eye(m)
    art_mask = (b < -eps).astype(float)
    A_full = np.hstack([A, S, A_art])
    c_full = np.hstack([np.zeros(n), np.zeros(m), -art_mask])
    basis = []
    for i in range(m):
        if b[i] >= -eps:
            basis.append(n + i)
        else:
            basis.append(n + m + i)
    all_idx = set(range(n + m + m))
    nbasis = [j for j in all_idx if j not in set(basis)]
    status, x_full, obj = PrimalSimplex(c_full, A_full, b, basis, nbasis)
    if status != "optimal" or obj < -eps:
        return "infeasible", None
    basis = list(basis)
    nbasis = [j for j in range(n + m + m) if j not in set(basis)]
    for i in range(m):
        jB = basis[i]
        if jB >= n + m:
            B = A_full[:, basis]
            try:
                Binv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                Binv = np.linalg.pinv(B)
            row = Binv[i, :] @ A_full
            found = False
            for j in list(nbasis):
                if j < n + m and abs(row[j]) > 1e-9:
                    basis[i] = j
                    nbasis.remove(j)
                    nbasis.append(jB)
                    found = True
                    break
            if not found:
                j_slack = n + i
                if j_slack in nbasis and abs(row[j_slack]) > 1e-9:
                    basis[i] = j_slack
                    nbasis.remove(j_slack)
                    nbasis.append(jB)
                else:
                    pass
    keep_cols = list(range(n + m))
    A2 = A_full[:, keep_cols]
    c2 = np.hstack([c, np.zeros(m)])
    new_basis = []
    for j in basis:
        if j < n + m:
            new_basis.append(j)
        else:
            new_basis.append(n + (j - (n + m)))
    used = set(new_basis)
    if len(new_basis) != m or len(used) != m:
        new_basis = []
        used = set()
        for i in range(m):
            j = basis[i]
            cand = j if j < n + m else n + i
            if cand in used:
                for s in range(n, n + m):
                    if s not in used:
                        cand = s
                        break
            new_basis.append(cand)
            used.add(cand)
    nbasis2 = [j for j in range(n + m) if j not in set(new_basis)]
    return "ok", (A2, c2, new_basis, nbasis2)

def Solve(c, A, b):
    if _unbounded_trivial(c, A):
        return "unbounded", None, None
    if np.all(b >= -eps):
        m, n = A.shape
        A2 = np.hstack([A, np.eye(m)])
        c2 = np.hstack([c, np.zeros(m)])
        return PrimalSimplex(c2, A2, b.astype(float))
    st, payload = Phase1_leq(c, A, b)
    if st == "infeasible":
        return "infeasible", None, None
    A2, c2, basis, nbasis = payload
    return PrimalSimplex(c2, A2, b.astype(float), basis, nbasis)

def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Dual Simplex method.")
    parser.add_argument("filename", type=str, help="Input file containing the LP problem.")
    return parser.parse_args()

def main():
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
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
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
    elif status == "infeasible":
        print("No solution found.")
    else:
        print("No solution found.")

if __name__ == '__main__':
    main()
