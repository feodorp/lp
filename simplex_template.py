import numpy as np
import argparse
from dataclasses import dataclass

Array = np.ndarray
eps_phase1 = 1e-9


def solve_upper_triangular(U, b, T):
    m = U.shape[0]
    x = b.copy()
    for i in range(m - 1, -1, -1):
        piv = U[i, i]
        if i < m - 1:
            if T:
                x[i] -= U[i + 1:, i] @ x[i + 1:]
            else:
                x[i] -= U[i, i + 1:] @ x[i + 1:]
        x[i] /= piv
    return x


def solve_lower_triangular(U, b, T):
    m = U.shape[0]
    x = b.copy()
    for i in range(m):
        if i > 0:
            if T:
                x[i] -= U[:i, i] @ x[:i]
            else:
                x[i] -= U[i, :i] @ x[:i]
        piv = U[i, i]
        x[i] /= piv
    return x


def lu(A):
    m, n = A.shape
    U = A.copy()
    L = np.eye(m, dtype=float)
    pvec = np.arange(m)

    for k in range(m):
        r = k + int(np.argmax(np.abs(U[k:, k])))
        if r != k:
            U[[k, r], :] = U[[r, k], :]
            if k > 0:
                L[[k, r], :k] = L[[r, k], :k]
            pvec[[k, r]] = pvec[[r, k]]

        for i in range(k + 1, m):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    return pvec, L, U


@dataclass
class LuFactors:
    pvec: Array
    L: Array
    U: Array
    eps: float = 1e-12

    @staticmethod
    def factor(B):
        pvec, L, U = lu(B)
        return LuFactors(pvec=pvec, L=L, U=U)

    def solve(self, b):
        y = solve_lower_triangular(self.L, np.asarray(b, float)[self.pvec], 0)
        return solve_upper_triangular(self.U, y, 0)

    def solve_t(self, b: Array) -> Array:
        z = solve_lower_triangular(self.U, b, 1)
        w = solve_upper_triangular(self.L, z, 1)
        inv = np.empty_like(self.pvec)
        inv[self.pvec] = np.arange(len(self.pvec))
        return np.asarray(w, float)[inv]

    def bg_replace_col(self, a_j, col_p):
        m = self.L.shape[0]

        v = solve_lower_triangular(self.L, np.asarray(a_j, float)[self.pvec], 0)

        H = self.U.copy()
        H[:, col_p] = v

        L_new = self.L.copy()
        for i in range(m - 1, col_p + 1, -1):
            piv = H[i-1, col_p]
            if abs(piv) < 1e-12:
                return False
            alpha = H[i, col_p] / piv
            if alpha != 0.0:
                H[i, col_p:] -= alpha * H[i-1, col_p:]
                H[i, col_p] = 0.0
                L_new[:, i - 1] += alpha * L_new[:, i]

        for i in range(col_p, m - 1):
            piv = H[i, i]
            if abs(piv) < 1e-12:
                return False
            alpha = H[i + 1, i] / piv
            if abs(alpha) > 0.0:
                H[i + 1, i:] -= alpha * H[i, i:]
                L_new[:, i] += alpha * L_new[:, i + 1]

        self.L = L_new
        self.U = H
        return True


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


def PrimalSimplex(c, A, b, basis=None, nbasis=None,
                  eps_rc=1e-9, eps_dir=1e-12, max_iter=100000):
    b = np.asarray(b, float).reshape(-1)
    c = np.asarray(c, float).reshape(-1)
    m, tot_n = A.shape
    n = tot_n - m

    if basis is None or nbasis is None:
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))
    else:
        basis = list(basis)
        nbasis = list(nbasis)

    B = A[:, basis]
    fac = LuFactors.factor(B)

    it = 0
    while True:
        it += 1
        if it > max_iter:
            raise RuntimeError("Too many iterations")

        # 1.
        bbar = fac.solve(b)

        # 2.
        cB = c[basis]
        y = fac.solve_t(cB)

        # 3.
        rN_vals = c[nbasis] - A[:, nbasis].T @ y
        rN_max = np.max(rN_vals)
        cand_js = []

        if rN_max > eps_rc:
            cand_js = [j for j, rj in zip(nbasis, rN_vals) if rj > eps_rc]

        if not cand_js:
            x_aug = np.zeros(tot_n, dtype=float)
            x_aug[basis] = bbar.copy()
            obj_full = float(c @ x_aug)
            x_star = x_aug[:n]
            return "optimal", x_star, obj_full

        # 4.
        entering = int(min(cand_js))
        a_j = A[:, entering]
        d = fac.solve(a_j)

        # 5.
        if np.all(d <= eps_dir):
            return "unbounded", None, None

        pos_rows = np.where(d > eps_dir)[0]
        theta_vals = bbar[pos_rows] / d[pos_rows]
        theta = theta_vals.min()

        # 6-7.
        tol = 1e-12
        near_min_rows = pos_rows[theta_vals <= theta + tol]
        leave_row = min(near_min_rows, key=lambda i: basis[i])
        leaving = basis[leave_row]

        basis[leave_row] = entering
        nbasis.remove(entering)
        nbasis.append(leaving)
        nbasis.sort()

        ok = fac.bg_replace_col(a_j, leave_row)
        if not ok:
            B = A[:, basis]
            fac = LuFactors.factor(B)


def Phase1(c, A, b, eps_phase1=1e-9):

    c_slack, A_eq, b, basis, nbasis = augment_with_slacks(c, A, b)
    m, n_tot = A_eq.shape

    ones = np.ones((m, 1))
    A_aux = np.hstack([A_eq, -ones])
    c_aux = np.concatenate([np.zeros(n_tot), np.array([-1.0])])
    x0_idx = n_tot

    if np.any(b < -eps_phase1):
        p = int(np.argmin(b))
        old = basis[p]
        basis[p] = x0_idx
        nbasis.append(old)
        nbasis.sort()

    status_aux, x_aux, obj_aux = PrimalSimplex(
        c_aux, A_aux, b, basis[:], nbasis[:],
        eps_rc=1e-12, eps_dir=1e-12
    )

    if status_aux != "optimal" or obj_aux < -eps_phase1:
        return "infeasible", None, None

    A2 = A_eq
    c2 = c_slack

    basis2 = basis[:]
    if x0_idx in basis2:
        p = basis2.index(x0_idx)
        candidates = [j for j in range(n_tot)
                      if (j not in basis2) and (abs(A2[p, j]) > eps_phase1)]
        if candidates:
            basis2[p] = min(candidates)
        else:
            j_enter = (n_tot - m) + p
            if j_enter in basis2:
                for j in range(n_tot):
                    if j not in basis2:
                        j_enter = j
                        break
            basis2[p] = j_enter

    nbasis2 = sorted(set(range(n_tot)) - set(basis2))

    status, x_star, _ = PrimalSimplex(
        c2, A2, b, basis2, nbasis2,
        eps_rc=1e-12, eps_dir=1e-12
    )
    if status == "optimal":
        obj = float(c @ x_star)
    else:
        obj = None
    return status, x_star, obj


def Solve(c, A, b):
    if np.all(b >= -1e-9):
        c_aug, A_eq, b_clean, basis, nbasis = augment_with_slacks(c, A, b)
        return PrimalSimplex(c_aug, A_eq, b_clean, basis, nbasis)
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
