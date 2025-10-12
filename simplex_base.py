import numpy as np
import sys
import argparse

from utils import pivot_operation

# global vars
eps = 0.00001

def PrimalSimplex(c, A, b, basis=None, nbasis=None, return_basis=False):
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
    else:
        assert len(basis) == m, f"Basis has {len(basis)} variables instead of {m}."
        assert len(nbasis) == n, f"Non-Basis has {len(nbasis)} variables instead of {n}."

    while True:
        idx_b, idx_n = np.array(basis), np.array(nbasis)
        B, N = A[:, idx_b], A[:, idx_n]
        c_b, c_n = c[idx_b], c[idx_n]

        # Подсказка: np.linalg.solve(M, v) решает систему Mx = v
        B_inv = np.linalg.inv(B)
        x_b = B_inv @ b

        # TODO: Посчитать reduced cost's
        y = B_inv.T @ c_b
        r_n = c_n - N.T @ y

        # TODO: Находим кандидата для входа в базис
        j = np.argmax(r_n)
        entering_index = idx_n[j]
        if r_n[j] < eps:
            break

        # TODO: Вычисляем направление, не забывая детектировать unbounded
        d = (B_inv @ A[:, entering_index].reshape(-1, 1)).ravel()  # partial derivatives
        if np.all(d < eps):
            return "unbounded", None, None

        # TODO: Найти кандидата для выхода из базиса
        d_pos_args = np.array([i for i in range(len(d)) if d[i] > eps])
        d_pos, x_b_pos = d[d_pos_args], x_b[d_pos_args]
        theta = x_b_pos / d_pos
        l = np.argmin(theta)
        leaving_index = idx_b[d_pos_args[l]]

        # TODO: Обновляем basis и nbasis
        basis, nbasis = pivot_operation(basis, nbasis, leaving_index, entering_index)

    # TODO: Восстановить исходную систему, восстановить x и вернуть результат
    z_star = c_b.T @ x_b
    x_star = np.zeros(n + m)
    x_star[idx_b] = x_b

    x_star_initial = x_star[:n]

    result = "optimal", x_star_initial, z_star
    if return_basis:
        return *result, basis, nbasis
    return result


def Phase1(c, A, b):
    m, n = A.shape

    # TODO: Создаем вспомогательную задачу
    # x_0 + initial variables + slacks -> 1 + m + n variables
    new_c = np.concatenate([np.array([-1]), np.zeros(m + n)])
    new_A = np.concatenate([np.ones((m, 1)), A, np.eye(m)], axis=1)
    basis = list(range(1 + n, 1 + n + m))
    nbasis = list(range(0, 1 + n))

    # force x_0 into basis
    leaving_index = np.argmin(b) + (1 + n)
    entering_index = 0

    basis, nbasis = pivot_operation(basis, nbasis, leaving_index, entering_index)

    status, _, obj, basis, nbasis = PrimalSimplex(new_c, new_A, b, basis, nbasis, return_basis=True)
    if status != "optimal" or obj > eps:
        return "infeasible", None, None

    # TODO: Нужно восстановить исходную задачу
    c = np.concatenate([c, np.zeros(m)])
    A = np.concatenate([A, np.eye(m)], axis=1)
    basis = np.array([i for i in basis if i != 0]) - 1
    nbasis = np.array([i for i in nbasis if i != 0]) - 1

    return PrimalSimplex(c, A, b, basis.tolist(), nbasis.tolist())

def Solve(c, A, b):
    if np.all(b >= 0):
        # TODO: Добавляем слаки в систему
        # initial variables + slacks
        m = A.shape[0]
        c = np.concatenate([c, np.zeros(m)])
        A = np.concatenate([A, np.eye(m)], axis=1)
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
