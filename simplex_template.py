import numpy as np
import sys
import argparse

# global vars
eps = 0.00001

def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    #  max c^T x
    #  Ax = b
    #  x >= 0
    #  n - число переменных
    #  m - число ограничений
    #  Да, действительно, считаем что n > m (с учетом слаков)
    # Предполагаем что проблема точно feasible но возможно unbounded
    m, n_total = A.shape
    n = n_total - m
    
    if basis is None or nbasis is None:
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))
    
    B = A[:, basis]
    B_inv = np.linalg.inv(B)
    x_B = B_inv @ b
    
    while True:
        c_B = c[basis]
        y = np.linalg.solve(B.T, c_B)
        c_N = c[nbasis]
        N = A[:, nbasis]
        reduced_cost = c_N - N.T @ y
        
        if np.all(reduced_cost <= eps):
            x = np.zeros(n_total)
            x[basis] = x_B
            return "optimal", x[:n], c[:n] @ x[:n]
        
        entering_idx_in_nb = np.argmax(reduced_cost)
        entering_idx = nbasis[entering_idx_in_nb]
        
        d = B_inv @ A[:, entering_idx]
        
        if np.all(d <= eps):
            return "unbounded", None, None
        
        ratios = np.full(m, np.inf)
        for i in range(m):
            if d[i] > eps:
                ratios[i] = x_B[i] / d[i]
        
        if np.all(ratios == np.inf):
            return "unbounded", None, None
        
        leaving_idx_in_b = np.argmin(ratios)
        leaving_idx = basis[leaving_idx_in_b]
        
        t = ratios[leaving_idx_in_b]
        x_B = x_B - t * d
        x_B[leaving_idx_in_b] = t
        
        basis[leaving_idx_in_b] = entering_idx
        nbasis[entering_idx_in_nb] = leaving_idx
        
        B = A[:, basis]
        B_inv = np.linalg.inv(B)


def Phase1(c, A, b):
    m, n_total = A.shape
    n = n_total - m
    
    new_c = np.zeros(n_total + m)
    new_c[n_total:] = -1
    
    new_A = np.zeros((m, n_total + m))
    new_A[:, :n_total] = A
    new_A[:, n_total:] = np.eye(m)
    
    basis = list(range(n_total, n_total + m))
    nbasis = list(range(0, n_total))
    
    status, x, obj = PrimalSimplex(new_c, new_A, b, basis, nbasis)
    
    if status != "optimal" or obj < -eps:
        return "infeasible", None, None
    
    basis = []
    for i in range(n_total):
        if x[i] > eps or i in basis:
            basis.append(i)
    
    while len(basis) < m:
        for i in range(n_total):
            if i not in basis:
                basis.append(i)
                break
    
    nbasis = [i for i in range(n_total) if i not in basis]
    
    return PrimalSimplex(c, A, b, basis, nbasis)


def Solve(c, A, b):
    m, n_total = A.shape
    n = n_total - m
    
    if np.all(b >= 0):
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))
        return PrimalSimplex(c, A, b, basis, nbasis)
    
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
    
    A_full = np.hstack([A, np.eye(m)])
    c_full = np.concatenate([c, np.zeros(m)])
    
    print("n =", n)
    print("m =", m)
    print("c =", c)
    print("A =", A)
    print("b =", b)

    print("Solving the linear program using the Primal Simplex method...\n")
    status, solution, objective = Solve(c_full, A_full, b)
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