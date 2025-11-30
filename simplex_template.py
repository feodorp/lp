import numpy as np
import argparse

eps = 1e-8

class LU:

    def __init__(self, B):
        self.B = B.copy()
        self.L = self.U = None
        try:
            m = len(B)
            perm = np.arange(m)
            L, U = np.eye(m, dtype=np.float64), np.array(B, dtype=np.float64)
            for k in range(m):
                s = k + int(np.argmax(np.abs(U[k:, k])))
                if s != k:
                    U[[k, s], :] = U[[s, k], :]
                    if k > 0:
                        L[[k, s], :k] = L[[s, k], :k]
                    perm[[k, s]] = perm[[s, k]]
                piv = U[k, k]
                if abs(piv) < eps:
                    raise RuntimeError("Singular matrix")
                for i in range(k + 1, m):
                    L[i, k] = U[i, k] / piv
                    U[i, k:] -= L[i, k] * U[k, k:]
                    U[i, k] = 0.0
            self.L, self.U = L, U
            self.perm = perm
        except:
            pass
    
    def solve(self, b):

        if self.L is None:
            return np.linalg.solve(self.B, b)
        m = len(b)
        z = b[self.perm].copy()
        for i in range(m):
            z[i+1:] -= self.L[i+1:, i] * z[i]
        x = z.copy()
        for i in range(m - 1, -1, -1):
            if abs(self.U[i, i]) < eps:
                return np.linalg.solve(self.B, b)
            x[i] /= self.U[i, i]
            x[:i] -= self.U[:i, i] * x[i]
        return x
    
    def solveT(self, b):

        if self.L is None:
            return np.linalg.solve(self.B.T, b)
        m = len(b)
        w = b.copy()
        for i in range(m):
            if abs(self.U[i, i]) < eps:
                return np.linalg.solve(self.B.T, b)
            w[i] /= self.U[i, i]
            w[i+1:] -= self.U[i, i+1:] * w[i]
        y = w.copy()
        for i in range(m - 1, -1, -1):
            y[i] -= np.dot(self.L[i+1:, i], y[i+1:])
        return y[np.argsort(self.perm)]
    
    def update(self, p, d):

        if self.L is None:
            return False
        m = len(d)
        v = self.U @ d
        L, H = self.L.copy(), self.U.copy()
        H[:, p] = v
        for k in range(m - 1, p + 1, -1):
            piv = H[k - 1, p]
            if abs(piv) < eps:
                return False
            alpha = H[k, p] / piv
            if abs(alpha) > 0.0:
                H[k, p:] -= alpha * H[k - 1, p:]
                L[:, k - 1] += alpha * L[:, k]
        for k in range(p, m - 1):
            piv = H[k, k]
            if abs(piv) < eps:
                return False
            alpha = H[k + 1, k] / piv
            if abs(alpha) > 0.0:
                H[k + 1, k:] -= alpha * H[k, k:]
                L[:, k] += alpha * L[:, k + 1]
        self.L, self.U = L, H
        self.B[:, p] = self.B[:, p]
        return True


def DualSimplex(c, A, b, max_iters=5000):

    m, n = A.shape
    
    A = np.hstack([A, np.eye(m)])
    c = np.concatenate([c, np.zeros(m)])
    total = n + m
    
    basis = list(range(n, total))
    nbasis = list(range(n))
    
    scale = max(1.0, np.max(np.abs(A)), np.max(np.abs(b)), np.max(np.abs(c)))
    U = np.full(total, 1e6 * scale)
    
    lu = LU(A[:, basis])
    
    iters = 0
    while iters < max_iters:
        iters += 1
        
        # dual y
        y = lu.solveT(c[basis])
        
        # get current \overline{C_N}
        N = A[:, nbasis]
        c_bar = c[nbasis] - N.T @ y

        # get current d_N and x_N
        d_N = np.where(c_bar > 0.0, -1, 1)
        x_N = np.where(c_bar > 0.0, U[nbasis], 0.0)
        r = d_N * c_bar # current residuals
        assert np.all(r <= eps), "dual is not feasible"

        # solve with current nonbasis
        x_B = lu.solve(b - N @ x_N)
        
        # residuals are feasible check only basis variables
        if np.all(x_B >= -eps):

            x = np.zeros(total)
            x[basis] = x_B
            x[nbasis] = x_N

            # if some of origin variables large it means task unbounded            
            if np.any(x[:n] >= U[:n] - eps):
                return "unbounded", None, None

            # return origin variables
            return "optimal", x[:n], float(np.dot(c[:n], x[:n]))
        

        # select violating variable for update
        leave_pos = np.flatnonzero(x_B < -eps)[0]
        leaving_idx = basis[leave_pos]
        
        # recompute sigma to keep r non positive
        e_i = np.zeros(m)
        e_i[leave_pos] = 1.0        
        sigma = -(lu.solveT(e_i) @ N) * d_N
        valid_mask = sigma > eps
        
        # all sigma are negative so dual is unbounded and primal is infeasible
        if not np.any(valid_mask):
            return "infeasible", None, None
        
        # find minimal ratio with positive sigma
        ratios = np.full(n, np.inf)
        ratios[valid_mask] = -r[valid_mask] / sigma[valid_mask]
        enter_pos = np.flatnonzero((ratios <= np.min(ratios) + eps) & valid_mask)[0]
        entering_idx = nbasis[enter_pos]
        
        basis[leave_pos] = entering_idx
        nbasis[enter_pos] = leaving_idx
        
        if not lu.update(leave_pos, lu.solve(A[:, entering_idx])) or iters % 50 == 0:
            lu = LU(A[:, basis])
    
    return "stalled", None, None


def Solve(c, A, b):
    """Solve max c^T x s.t. Ax <= b, x >= 0"""
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
