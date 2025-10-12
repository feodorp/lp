import numpy as np
import sys
import argparse

# global vars
eps = 0.00001

def lu_decomposition(A):
    # LU разложение
    
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        # Верхне треугольная
        for j in range(i, n):
            U[i, j] = A[i, j] - np.sum(L[i, :i] * U[:i, j])
        # Нижне треугольная
        for j in range(i, n):
            if i == j:
                L[i, i] = 1.0
            else:
                L[j, i] = (A[j, i] - np.sum(L[j, :i] * U[:i, i])) / U[i, i] if abs(U[i, i]) > eps else 0.0
    
    return L, U


def solve_lu(L, U, b):
    # Решение СЛАУ с помощью LU

    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.sum(L[i, :i] * y[:i])
    
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.sum(U[i, i+1:] * x[i+1:])) / U[i, i] if abs(U[i, i]) > eps else 0.0
    
    return x

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

    B = A[:, basis]
    L, U = lu_decomposition(B)
    inv_B = np.array([solve_lu(L, U, np.eye(m)[:, i]) for i in range(m)]).T

    while True:
        # Подсказка: np.linalg.solve(M, v) решает систему Mx = v
        c_B = c[basis]
        y = inv_B.T @ c_B
        A_N = A[:, nbasis]
        c_N = c[nbasis]

        # TODO: Посчитать reduced cost's я
        reduced_cost = c_N - A_N.T @ y

        pos = np.where(reduced_cost > eps)[0]
        if len(pos) == 0:
            break

        # TODO: Находим кандидата для входа в базис
        candidates = [nbasis[k] for k in pos]
        entering_index = min(candidates)

        # TODO: Вычисляем направление, не забывая детектировать unbounded
        a_j = A[:, entering_index]
        d = inv_B @ a_j
        pos_d = np.where(d > eps)[0]
        if len(pos_d) == 0:
            return "unbounded", None, None

        # TODO: Найти кандидата для выхода из базиса
        x_B = inv_B @ b
        ratios = np.full(m, np.inf)
        ratios[pos_d] = x_B[pos_d] / d[pos_d]
        leaving_pos = np.argmin(ratios)
        leaving_index = basis[leaving_pos]
        
        # TODO: Обновляем basis и nbasis
        basis[leaving_pos] = entering_index
        nbasis.remove(entering_index)
        nbasis.append(leaving_index)

        # 
        p = leaving_pos
        a_new = A[:, entering_index]
        a_old = A[:, leaving_index]
        B_new = A[:, basis]
        L_new, U_new = lu_decomposition(B_new)
        inv_B = np.array([solve_lu(L_new, U_new, np.eye(m)[:, i]) for i in range(m)]).T

    # TODO: Восстановить исходную систему, восстановить x и вернуть результат
    x_B = inv_B @ b
    c_B = c[basis]
    obj = c_B @ x_B
    x = np.zeros(n + m)
    x[basis] = x_B
    return "optimal", x[:n], obj


def Phase1(c, A, b):
    # TODO: Создаем вспомогательную задачу
    original_n = len(c)
    m_len = len(b)
    new_A = np.hstack((A, np.eye(m_len), -np.ones((m_len, 1))))
    new_c = np.hstack((np.zeros(original_n + m_len), -1.0))
    x0_idx = original_n + m_len
    basis = list(range(original_n, original_n + m_len))
    p = np.argmin(b)
    basis[p] = x0_idx
    nbasis = [i for i in range(original_n + m_len + 1) if i not in basis]

    status, x, obj = PrimalSimplex(new_c, new_A, b, basis, nbasis)
    if status != "optimal" or obj < -eps:
        return "infeasible", None, None

    status, x, obj = PrimalSimplex(new_c, new_A, b, basis, nbasis)
    if status != "optimal" or obj > eps:
        return "infeasible", None, None
    
    # TODO: Нужно восстановить исходную задачу
    if x0_idx in basis:
        B = new_A[:, basis]
        p = basis.index(x0_idx)
        L, U = lu_decomposition(B)
        inv_B_phase = np.array([solve_lu(L, U, np.eye(m_len)[:, i]) for i in range(m_len)]).T
        found = False
        for ent in nbasis:
            a_j = new_A[:, ent]
            d = inv_B_phase @ a_j
            if d[p] > eps:
                basis[p] = ent
                nbasis.remove(ent)
                nbasis.append(x0_idx)
                found = True
                break
    
    c = np.hstack((c, np.zeros(m_len)))
    A = np.hstack((A, np.eye(m_len)))
    # basis = basis
    nbasis = [i for i in nbasis if i != x0_idx]

    return PrimalSimplex(c, A, b, basis, nbasis)


def Solve(c, A, b):
    if np.all(b >= 0):
        # TODO: Добавляем слаки в систему
        A = np.hstack((A, np.eye(len(b))))
        c = np.hstack((c, np.zeros(len(b))))
        return PrimalSimplex(c, A, b)
    
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