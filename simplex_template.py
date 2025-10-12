import numpy as np
import sys
import argparse

eps = 1e-9

def lu_decompose(A):
    """
    Performs LU decomposition of a square matrix A.
    Returns L, U, P such that PA = LU.
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)

    for k in range(n - 1):
        pivot_row = k + np.argmax(np.abs(U[k:, k]))
        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        if abs(U[k, k]) < eps:
            raise ValueError("Matrix is singular, LU decomposition not possible without further pivoting strategies.")

        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    return L, U, P

def solve_lu(L, U, P, b):
    """
    Solves the linear system Ax = b using LU decomposition PA = LU.
    """
    Pb = P @ b
    y = np.zeros_like(Pb, dtype=float)
    for i in range(len(L)):
        y[i] = (Pb[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    x = np.zeros_like(y, dtype=float)
    for i in range(len(U) - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    """
     max c^T x
     Ax = b
     x >= 0
     n - число переменных
     m - число ограничений
     Да, действительно, считаем что n > m (с учетом слаков)
    Предполагаем что проблема точно feasible но возможно unbounded
    """
    m, n = A.shape

    cur_iter = 0
    max_iter = 1000

    current_basis = list(basis)
    current_nbasis = list(nbasis)

    while cur_iter < max_iter:
        cur_iter += 1

        B_matrix = A[:, current_basis]
        N_matrix = A[:, current_nbasis]

        # Solve B * x_B = b
        try:
            L_B, U_B, P_B = lu_decompose(B_matrix)
            x_B = solve_lu(L_B, U_B, P_B, b)
        except ValueError:
            return "infeasible", None, None

        # Check for feasibility of basic solution
        if np.any(x_B < -eps):
            return "infeasible", None, None

        # Solve B.T * y = c_B
        c_B = c[current_basis]
        try:
            L_BT, U_BT, P_BT = lu_decompose(B_matrix.T)
            y = solve_lu(L_BT, U_BT, P_BT, c_B)
        except ValueError:
            return "infeasible", None, None

        c_N = c[current_nbasis]
        reduced_costs = c_N - N_matrix.T @ y

        entering_candidate_idx = np.argmax(reduced_costs)
        entering_cost = reduced_costs[entering_candidate_idx]

        if entering_cost < eps:
            break

        entering_var_idx_global = current_nbasis[entering_candidate_idx]
        a_entering = A[:, entering_var_idx_global]

        # Solve B * d = a_entering
        try:
            d = solve_lu(L_B, U_B, P_B, a_entering)
        except ValueError:
            return "infeasible", None, None

        if np.all(d <= eps):
            return "unbounded", None, None

        ratios = []
        for i in range(m):
            if d[i] > eps:
                ratios.append(x_B[i] / d[i])
            else:
                ratios.append(np.inf)
        
        leaving_basic_idx_in_basis = np.argmin(ratios)
        theta_star = ratios[leaving_basic_idx_in_basis]

        if theta_star == np.inf:
            return "unbounded", None, None

        leaving_var_idx_global = current_basis[leaving_basic_idx_in_basis]

        current_basis[leaving_basic_idx_in_basis] = entering_var_idx_global
        current_nbasis[entering_candidate_idx] = leaving_var_idx_global

    if cur_iter == max_iter:
        return "stalling", None, None

    x_star_full = np.zeros(n)
    x_star_full[current_basis] = x_B

    objective_value = c @ x_star_full

    return "optimal", x_star_full, objective_value

def Phase1(c, A, b):
    m, n = A.shape

    if np.all(b >= -eps):
        A_std = np.hstack((A, np.eye(m)))
        c_std = np.zeros(n + m)
        c_std[:n] = c
        basis = list(range(n, n + m))
        nbasis = list(range(n))
        return PrimalSimplex(c_std, A_std, b, basis, nbasis)

    new_n = n + m + 1
    new_c = np.zeros(new_n)
    new_c[-1] = -1.0
    new_A = np.hstack((A, np.eye(m), -np.ones((m, 1))))
    basis = list(range(n, n + m))
    nbasis = list(range(n)) + [n + m]

    p_row = np.argmin(b)
    s_p_idx_in_basis = basis.index(n + p_row)
    basis[s_p_idx_in_basis] = n + m
    nbasis.append(n + p_row)
    nbasis.remove(n + m)

    new_status, new_x_star, new_obj = PrimalSimplex(new_c, new_A, b, basis, nbasis)

    if new_status == "unbounded":
        return "infeasible", None, None

    if abs(new_obj) > eps: 
        return "infeasible", None, None

    x0_global_idx = n + m
    basis_phase1 = [idx for idx in basis if idx != x0_global_idx]
    nbasis_phase1 = [idx for idx in nbasis if idx != x0_global_idx]

    A_phase2 = new_A[:, :new_n - 1]
    c_phase2 = np.zeros(n + m)
    c_phase2[:n] = c

    if len(basis_phase1) < m:
        return "infeasible", None, None

    return PrimalSimplex(c_phase2, A_phase2, b, basis_phase1, nbasis_phase1, n)

def Solve(c, A, b):
    m, n = A.shape
    A_std = np.hstack((A, np.eye(m)))
    c_std = np.zeros(n + m)
    c_std[:n] = c

    if np.all(b >= -eps):
        # Добавляем слаки в систему
        basis = list(range(n, n + m))
        nbasis = list(range(n))
        return PrimalSimplex(c_std, A_std, b, basis, nbasis)
    
    # Иначе запускаем фазу 1
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
