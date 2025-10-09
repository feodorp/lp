import random

import numpy as np
import argparse

# global vars
eps = 1e-8

class LU:
    
    def __init__(self, B):
        self.B = B.copy()
        self.L = self.U = None
        try:
            m = len(B)
            perm = np.arange(m)
            L, U = np.eye(m, dtype=np.float32), np.array(B, dtype=np.float32)
            for k in range(m):
                s = k + int(np.argmax(np.abs(U[k:, k])))
                if s != k:
                    U[[k, s], :] = U[[s, k], :]
                    if k > 0:
                        L[[k, s], :k] = L[[s, k], :k]
                    perm[[k, s]] = perm[[s, k]]
                piv = U[k, k]
                if abs(piv) < eps:
                    raise RuntimeError("Cannot make LU for B")
                for i in range(k + 1, m):
                    L[i, k] = U[i, k] / piv
                    U[i, k:] -= L[i, k] * U[k, k:]
                    U[i, k] = 0.0
            self.L, self.U = L, U
            self.perm = perm
        except:
            print("Warning: LU failed to convert B to LU so using numpy solve")
    
    def solve(self, b):
        
        if self.L is None:
            return np.linalg.solve(self.B, b)
        
        # Lz = b
        m = len(b)
        z = b[self.perm].copy()
        for i in range(m):
            z[i+1:] -= self.L[i+1:, i] * z[i]
        # Ux = z
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
        
        # U^T w = b
        m = len(b)
        w = b.copy()
        for i in range(m):
            if abs(self.U[i, i]) < eps:
                return np.linalg.solve(self.B, b)
            w[i] /= self.U[i, i]
            w[i+1:] -= self.U[i, i+1:] * w[i]
        # L^T y = w
        y = w.copy()
        for i in range(m - 1, -1, -1):
            y[i] -= np.dot(self.L[i+1:, i], y[i+1:])
            
        return y[np.argsort(self.perm)]
    
    def update(self, p, d):
        
        if self.L is None:
            return False
        
        # v = L^{-1} col
        m = len(d)
        v = self.U @ d
        
        L, H = self.L.copy(), self.U.copy()
        for k in range(m - 1, p + 1, -1):
            piv = H[k - 1, p]
            if abs(piv) < eps:
                return False
            alpha = H[k-1, p] / piv
            if abs(alpha) > 0.0:
                H[k, p:] -= alpha * H[k - 1, p:]
                L[:, k - 1] += alpha * L[:, k]

        H[:, p] = v
        for k in range(p, m-1):
            piv = H[k, k]
            if abs(piv) < eps:
                return False
            alpha = H[k + 1, k] / piv
            if abs(alpha) > 0.0:
                H[k+1, k:] -= alpha * H[k, k:]
                L[:, k] += alpha * L[:, k + 1]
        self.L, self.U = L, H

        return True

def PrimalSimplex(c, A, b, basis=None, nbasis=None):
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
    
    lu = LU(A[:, basis])
    
    iters = 0
    while True:
        iters += 1
        
        B, N = A[:, basis], A[:, nbasis]
        
        # c_{N} - A_{N}^T B^{T}^{-1} c_{B}
        # reduced_cost = c[nbasis] - N.T @ np.linalg.solve(B.T, c[basis])
        reduced_cost = c[nbasis] - N.T @ lu.solveT(c[basis])
        
        # can we increase the objective?
        pos_indexes = np.where(reduced_cost > eps)[0]
        if len(pos_indexes) == 0:
            
            # set current basis variables
            x = np.zeros(A.shape[1])
            # x[basis] = np.linalg.solve(B, b)
            x[basis] = lu.solve(b)
            
            # return first n variables, remained are slacks
            return "optimal", x[:n], np.dot(c, x)
        
        # take first positive item by Bland rule
        j = int(pos_indexes[0])
        entering_index = nbasis[j]
        
        # d = B^{-1}a_j
        # d = np.linalg.solve(B, N[:, j])
        d = lu.solve(N[:, j])
        mask = d > eps
        if np.all(~mask):
            return "unbounded", None, None

        # x_B' = B^{-1}b - d t*
        # t* = x_{B,i} / d_i
        # xb = np.linalg.solve(B, b)
        xb = lu.solve(b)
        r = np.where(mask)[0][np.argmin(xb[mask] / d[mask])]
        leaving_index = basis[r]
        
        # B - i + j
        # N - j + i
        basis[r] = entering_index
        nbasis[j] = leaving_index
        
        if not lu.update(r, d) or iters % 30 == 0:
            lu = LU(A[:, basis])
    

def Phase1(c, A, b):
    
    # Ax - Ix_{n+m} = b
    m, n = A.shape
    _c = np.concatenate([
        np.zeros(n, dtype=np.float32),
        np.zeros(m, dtype=np.float32),
        -np.ones(1, dtype=np.float32)
    ])
    _A = np.concatenate([
        A, np.eye(m, dtype=np.float32),
        -np.ones((m, 1), dtype=np.float32)
    ], axis=-1)
    basis = list(range(n, n + m))
    nbasis = list(range(0, n)) + [n + m]
    
    # one pivot iteration to put x_{n+m} into basis
    j = int(np.argmin(b))
    basis[j] = n + m
    nbasis[-1] = n + j
    # now xB >= 0 and all conditions must hold
    assert np.all(np.linalg.solve(_A[:, basis], b) >= -eps)
    
    # solving subtask
    status, _, obj = PrimalSimplex(_c, _A, b, basis, nbasis)
    if status != "optimal" or obj < -eps:
        return "infeasible", None, None
    
    if n + m in basis:
        # okay if x_{n+m} still in basis then it's value must be near eps because we minimized x_{n+m}
        # x_{n+m} = {B^{-1}b}_{n+m} - (B^{-1}A_{N}x_{N})_{n+m} = 0
        # x_{n+m} = {B^{-1}b}_{n+m} = 0
        # x_{n+m} = e_{n+m}B^{-1}A_{N}x_{N} = 0
        # \forall j t = x_{n+m} / (d_j = e_{n+m}^{T}B^{-1}a_j) = 0
        # so for any nonzero e_{n+m}B^{-1}a_j we can step by 0 in direction of x_j
        r = basis.index(n + m)
        e_nm = np.zeros(len(basis), dtype=np.float32)
        e_nm[r] = 1.0
        u = np.linalg.solve(_A[:, basis].T, e_nm)
        d = u.T @ _A[:, nbasis]
        j = nbasis[np.where(np.abs(d) > 0.0)[0][0]]
        basis[r] = j
        nbasis.remove(j)
    else:
        # remove extra variable
        nbasis.remove(n + m)
    
    # add slacks to base task
    c = np.concatenate([c, np.zeros(m, dtype=np.float32)])
    A = np.concatenate([A, np.eye(m, dtype=np.float32)], axis=-1)
    # check that found basis is feasible
    if np.any(np.linalg.solve(A[:, basis], b) < -eps):
        return "infeasible", None, None
    
    return PrimalSimplex(c, A, b, basis, nbasis)

def take_independent_conditions(A):
    rows = []
    for row_idx in range(len(A)):
        if np.linalg.matrix_rank(A[rows + [row_idx], :]) == len(rows) + 1:
            rows.append(row_idx)
    return np.array(rows, dtype=np.int32)

def Solve(c, A, b):
    
    # greedy taking independent rows
    # rows = take_independent_conditions(A)
    # A, b = A[rows], b[rows]
    
    # check if simplest solution is feasible
    if np.all(b >= 0):
        m, n = A.shape
        A = np.concatenate([A, np.eye(m)], axis=1)
        c = np.concatenate([c, np.zeros(m)], axis=0)
        return PrimalSimplex(c, A, b)
    
    # trying to find first feasible solution with solving subtask
    return Phase1(c, A, b)

def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Primal Simplex method.")
    parser.add_argument("--filename", type=str, help="Input file containing the LP problem.")
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