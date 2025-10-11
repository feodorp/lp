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
    m, n = A.shape
    n -= m
    if basis is None or nbasis is None:
        basis = list(range(n, n + m))
        nbasis = list(range(0, n))

    tmp = 0
    while tmp < 100:
        tmp += 1

        B = A[:, basis]
        N = A[:, nbasis]
        cB = c[basis]
        cN = c[nbasis]

        try:
            bbar = np.linalg.solve(B, b)
            y = np.linalg.solve(B.T, cB)
            rN = cN - (N.T).dot(y)
            z0 = cB.T.dot(bbar)
        except:
            return "infeasible", None, None

        # TODO: Посчитать reduced cost's 
        reduced_cost = rN

        # Проверка на оптимальность решения
        if np.all(reduced_cost <= eps):
            x = np.zeros(n + m)
            for i, bi in enumerate(basis):
                x[bi] = bbar[i]
            obj = c.dot(x)
            return "optimal", x, obj

        # TODO: Находим кандидата для входа в базис
        # Использую правило Блэнда
        entering_index = 0
        for i, val in enumerate(reduced_cost):
            if val > eps:
                entering_index = i
                break
        

        # TODO: Вычисляем направление, не забывая детектировать unbounded
        aj = A[:, entering_index]
        d = np.linalg.solve(B, aj)

        if np.all(d <= eps):
            return "unbounded", None, None

        # TODO: Найти кандидата для выхода из базиса
        min_i = 0
        min_rat = np.inf
        for i, d_i in enumerate(d):
            if d_i > eps:
                rat = bbar[i] / d_i
                if rat < min_rat:
                    min_rat = rat
                    min_i = i

        leaving_index = min_i
        
        
        # TODO: Обновляем basis и nbasis
        entering_var = nbasis[entering_index]
        leaving_var = basis[leaving_index]

        basis[leaving_index] = entering_var
        nbasis[entering_index] = leaving_var

    # TODO: Восстановить исходную систему, восстановить x и вернуть результат
    # (сделано в теле while)
    return "iteration limit", None, None


def Phase1(c, A, b):
    m, n = A.shape
    n -= m
    # TODO: Создаем вспомогательную задачу
    new_c = np.zeros(n + m + 1)
    new_c[0] = -1
    new_A = np.zeros((m, n + m + 1))
    new_A[:, 1::] = A
    new_A[:, 0] = -1
    
    min_i = 0
    min_b = np.inf
    for i, b_i in enumerate(b):
        if b_i < min_b:
            min_b = b_i
            min_i = i

    p = min_i

    basis = list(range(n + 1, n + m + 1))
    nbasis = list(range(0, n + 1))

    entering_var = nbasis[0]
    leaving_var = basis[p]

    basis[p] = entering_var
    nbasis[0] = leaving_var
    

    status, x, obj = PrimalSimplex(new_c, new_A, b, basis, nbasis)
    if status != "optimal" or obj > eps:
        return "infeasible", None, None
    
    # TODO: Нужно восстановить исходную задачу
    basis = [idx - 1 for idx in basis if idx != 0]
    nbasis = [idx - 1 for idx in nbasis if idx != 0]

    return PrimalSimplex(c, A, b, basis, nbasis)


def Solve(c, A, b):
    m, n = A.shape

    A = np.hstack([A, np.eye(m)])
    c = np.hstack([c, np.zeros(m)])

    if np.all(b >= 0):
        return PrimalSimplex(c, A, b)
    
    # Иначе запускаем фазу 1
    return Phase1(c, A, b)

def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Primal Simplex method.")
    parser.add_argument("filename", type=str, help="Input file containing the LP problem.")
    return parser.parse_args()

def main():
    # boilerplate for reading input data

    # args = proc_cmd()
    # with open(args.filename, 'r', encoding='utf-8') as f:
    with open("example_phase1.txt", 'r', encoding='utf-8') as f:
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