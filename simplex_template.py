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

def Phase1(c, A, b):
    # TODO: Создаем вспомогательную задачу
    m, n = A.shape
    new_c = np.concatenate([np.zeros(m + n), -np.ones(1)])
    new_A = np.concatenate([A, np.eye(m), -np.ones(m)[:, None]], axis=1)
    basis = list(range(n, n + m + 1))
    nbasis = list(range(0, n))
    p = np.argmin(b)
    basis.remove(p + n)
    nbasis.append(p + n)
    status, x, obj = PrimalSimplex(new_c, new_A, b, basis, nbasis)
    if status != "optimal" or obj > eps:
        return "infeasible", None, None
    
    # TODO: Нужно восстановить исходную задачу
    c = np.concatenate([c, np.zeros(m)])
    A = np.concatenate([A, np.eye(m)], axis=1)
    if m + n in basis:
        basis.remove(m + n)
    if m + n in nbasis:
        nbasis.remove(m + n)

    return PrimalSimplex(c, A, b, basis, nbasis)


def Solve(c, A, b):
    if np.all(b >= 0):
        # TODO: Добавляем слаки в систему
        m, n = A.shape
        a, b, c = PrimalSimplex(np.concatenate([c, np.zeros(m)]), np.concatenate([A, np.eye(m)], axis=1), b)
        if b is not None:
            b = b[:n]
        return a, b, c
    
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