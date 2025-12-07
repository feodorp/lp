import numpy as np
import sys
import argparse
import scipy
# global vars
eps = 0.00001


def lu_decomposition_pivot(A):
    n = len(A)
    P = np.eye(n)
    L = np.eye(n)
    U = A.copy()
    for k in range(n - 1):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if pivot != k:
            U[[k, pivot]] = U[[pivot, k]]
            P[[k, pivot]] = P[[pivot, k]]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P.T, L, U
class UpdatableMatrix:
    def __init__(self, A, eps):
        self.P, self.L, self.U = lu_decomposition_pivot(A)
        self.eps = eps
        self.n = A.shape[0]
        self.G = np.eye(self.n)
        self.G_inv = np.eye(self.n)
        # self.G_inv_t = np.eye(self.n)
    def update_column(self, idx, values: np.ndarray):
        values = values.reshape(-1, 1)
        v = self.G_inv @ scipy.linalg.solve_triangular(self.L, self.P.T @ values, lower=True)
        self.U[:, idx] = v.flatten()
        p = self.n - 1
        while p > idx and abs(self.U[p][idx]) < self.eps:
            p -= 1
        transitions = []
        for i in range(idx + 1, p):
            coef = self.U[i][idx] / self.U[p][idx]
            self.U[i] -= coef * self.U[p]
            transitions.append((i, coef, p))
        if p != idx:
            if abs(self.U[idx][idx]) < self.eps:
                self.U[idx] -= self.U[p]
                transitions.append((idx, 1.0, p))
            coef = self.U[p][idx] / self.U[idx][idx]
            self.U[p] -= coef * self.U[idx]
            transitions.append((p, coef, idx))
            for i in range(idx + 1, p):
                if abs(self.U[p][i]) < self.eps:
                    continue
                coef = self.U[p][i] / self.U[i][i]
                self.U[p] -= coef * self.U[i]
                transitions.append((p, coef, i))
        for i, coef, j in transitions:
            self.G[:, j] += coef * self.G[:, i]
            self.G_inv[i] -= coef * self.G_inv[j]

    def solve(self, b: np.ndarray, transpose=False):
        b = b.flatten()
        if not transpose:
            b = self.P.T @ b
            b = scipy.linalg.solve_triangular(self.L, b, lower=True)
            b = self.G_inv @ b
            b = scipy.linalg.solve_triangular(self.U, b, lower=False)
        else:
            b = scipy.linalg.solve_triangular(self.U.T, b, lower=True)
            b = self.G_inv.T @ b
            b = scipy.linalg.solve_triangular(self.L.T, b, lower=False)
            b = self.P @ b
        return b




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
    B = UpdatableMatrix(A[:, basis], eps)
    N = A[:, nbasis]
    while True:
        # Подсказка: np.linalg.solve(M, v) решает систему Mx = v

        # TODO: Посчитать reduced cost's 
        y = B.solve(c[basis], transpose=True)
        reduced_cost = c[nbasis] - (N.T @ y).reshape(-1)

        # TODO: Находим кандидата для входа в базис
        if reduced_cost.max() <= eps:
            break
        entering_index = np.random.choice(np.arange(n)[reduced_cost > eps])
        entering_index = np.argmax(reduced_cost)
        # TODO: Вычисляем направление, не забывая детектировать unbounded
        d = B.solve(A[:, nbasis[entering_index]])
        if d.max() <= eps:
            return "unbounded", None, None

        # TODO: Найти кандидата для выхода из базиса
        leaving_index = np.argmin(B.solve(b)[d > eps] / d[d > eps])
        leaving_index = np.arange(m)[d > eps][leaving_index]
        # TODO: Обновляем basis и nbasis
        B.update_column(leaving_index, N[:, entering_index])
        basis[leaving_index], nbasis[entering_index] = nbasis[entering_index], basis[leaving_index]
        N = A[:, nbasis]

    # TODO: Восстановить исходную систему, восстановить x и вернуть результат
    x = np.zeros(n + m)
    x[basis] = B.solve(b)
    return "optimal", x, c.T @ x

def DualSimplex(c, A, b):
    m, n = A.shape
    A_full = np.concatenate([A, np.eye(m)], axis=1)
    c_full = np.concatenate([c, np.zeros(m)])
    basis = list(range(n, n + m))
    nbasis = list(range(0, n))

    B = UpdatableMatrix(A_full[:, basis], eps)
    N = A_full[:, nbasis]

    y = B.solve(c_full[basis], transpose=True)
    cbarN = c_full[nbasis] - (N.T @ y).reshape(-1)

    U = 1e12
    dj = np.where(cbarN <= 0, 1, -1)
    xN = np.where(dj == 1, 0.0, U)

    xB = B.solve(b - N @ xN)

    r = dj * cbarN
    while True:
        if np.all(xB >= -eps):
            status2, x_final, obj = PrimalSimplex(c_full, A_full, b, basis=basis, nbasis=nbasis)

            if status2 == "optimal":
                return "optimal", x_final[:n], obj
            else:
                return status2, None, None

        i_local = np.argmin(xB)
        a_hat = np.zeros((m, len(nbasis)))
        for j_idx, col in enumerate(nbasis):
            a_hat[:, j_idx] = B.solve(A_full[:, col])

        a_hat_row = a_hat[i_local, :]
        sigma = - a_hat_row * dj

        candidates = np.where(sigma > eps)[0]
        if candidates.size == 0:
            return "infeasible", None, None
        ratios = -r[candidates] / sigma[candidates]
        valid_idxs = np.where(ratios >= -eps)[0]
        if valid_idxs.size == 0:
            return "infeasible", None, None

        min_pos = np.argmin(ratios[valid_idxs])
        chosen_idx_in_candidates = candidates[valid_idxs[min_pos]]
        t = ratios[valid_idxs[min_pos]]
        jstar_pos = chosen_idx_in_candidates
        jstar_var = nbasis[jstar_pos]
        r = r + sigma * t
        xB = xB - a_hat[:, jstar_pos] * dj[jstar_pos] * t
        B.update_column(i_local, A_full[:, jstar_var])
        leaving_var = basis[i_local]
        basis[i_local] = jstar_var
        nbasis[jstar_pos] = leaving_var

        N = A_full[:, nbasis]
        y = B.solve(c_full[basis], transpose=True)
        cbarN = c_full[nbasis] - (N.T @ y).reshape(-1)
        dj = np.where(cbarN <= 0, 1, -1)
        xN = np.where(dj == 1, 0.0, U)
        xB = B.solve(b - N @ xN)
        r = dj * cbarN

def Solve(c, A, b):
    if np.all(b >= 0):
        # TODO: Добавляем слаки в систему
        m, n = A.shape
        a, b, c = PrimalSimplex(np.concatenate([c, np.zeros(m)]), np.concatenate([A, np.eye(m)], axis=1), b)
        if b is not None:
            b = b[:n]
        return a, b, c
    
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