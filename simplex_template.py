import numpy as np
import argparse

eps = 1e-9


def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    """
    Прямой симплекс-метод.
    Решает задачу:
        max c^T x
        Ax = b
        x >= 0
    """
    m, n = A.shape

    if basis is None:
        basis = list(range(n - m, n))
    if nbasis is None:
        nbasis = list(range(0, n - m))

    while True:
        # Матрица базисных столбцов
        B = A[:, basis]
        # Решаем Bx_B = b
        x_B = np.linalg.solve(B, b)
        # Коэффициенты при небазисных переменных
        N = A[:, nbasis]
        # Стоимость базиса
        c_B = c[basis]
        c_N = c[nbasis]
        # Вектор потенциалов
        y = np.linalg.solve(B.T, c_B)
        # Вычисляем приведённые стоимости
        reduced_costs = c_N - N.T @ y

        # Проверка оптимальности
        if np.all(reduced_costs <= eps):
            x = np.zeros(n)
            x[basis] = x_B
            return "optimal", x, c @ x

        # Выбираем входящую переменную
        j = np.argmax(reduced_costs)
        if reduced_costs[j] <= eps:
            return "optimal", x_B, c @ x_B
        entering = nbasis[j]

        # Направление движения
        d = np.linalg.solve(B, A[:, entering])
        if np.all(d <= eps):
            return "unbounded", None, None

        # Определяем минимальное отношение
        ratios = [x_B[i] / d[i] if d[i] > eps else np.inf for i in range(m)]
        leaving_idx = np.argmin(ratios)
        leaving = basis[leaving_idx]

        # Обновляем базис
        basis[leaving_idx] = entering
        nbasis[j] = leaving


def Phase1(c, A, b):
    """
    Фаза 1: поиск допустимого решения.
    """
    m, n = A.shape
    # Добавляем искусственные переменные
    A_ext = np.hstack([A, np.eye(m)])
    c_ext = np.hstack([np.zeros(n), -np.ones(m)])
    basis = list(range(n, n + m))
    nbasis = list(range(0, n))
    status, x, obj = PrimalSimplex(c_ext, A_ext, b, basis, nbasis)

    if status != "optimal" or abs(obj) > eps:
        return "infeasible", None, None

    # Удаляем искусственные переменные
    x = x[:n]
    return PrimalSimplex(c, A, b)


def Solve(c, A, b):
    """
    Решает задачу линейного программирования.
    """
    if np.all(b >= 0):
        # Добавляем слаки
        A = np.hstack([A, np.eye(len(b))])
        c = np.hstack([c, np.zeros(len(b))])
        return PrimalSimplex(c, A, b)
    else:
        return Phase1(c, A, b)


def main():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Primal Simplex method.")
    parser.add_argument("filename", type=str, help="Input file containing the LP problem.")
    args = parser.parse_args()

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

    # ⬇️ Выводим входные данные
    print("===== INPUT DATA =====")
    print(f"n = {n}")
    print(f"m = {m}")
    print(f"c = {c}")
    print("A =")
    print(A)
    print(f"b = {b}")
    print("======================\n")

    print("Solving the linear program using the Primal Simplex method...\n")
    status, solution, objective = Solve(c, A, b)

    print("\n===== RESULT =====")
    print("Status:", status)
    if status == "optimal":
        print("Optimal solution x* =", solution)
        print("Optimal value =", objective)
    elif status == "unbounded":
        print("The problem is unbounded.")
    else:
        print("No feasible solution.")
    print("===================")


if __name__ == "__main__":
    main()

