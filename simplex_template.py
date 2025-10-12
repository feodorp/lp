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
    n = n_total - m  # исходные переменные без слаков

    if basis is None or nbasis is None:
        basis = list(range(n, n_total))
        nbasis = list(range(0, n))

    iter_count = 0
    max_iters = 1000  # защита от зацикливания

    while iter_count < max_iters:
        iter_count += 1

        # 1. Формируем базисную матрицу B
        B = A[:, basis]

        try:
            # 2. Решаем B * x_B = b для получения текущего базисного решения
            x_B = np.linalg.solve(B, b)

            # 3. Вычисляем симплекс-множители y = B^{-T} * c_B
            c_B = c[basis]
            y = np.linalg.solve(B.T, c_B)

            # 4. Вычисляем приведенные стоимости для небазисных переменных
            reduced_cost = np.zeros(len(nbasis))
            for j, nb_idx in enumerate(nbasis):
                reduced_cost[j] = c[nb_idx] - np.dot(A[:, nb_idx], y)

            # 5. Проверка на оптимальность
            if np.all(reduced_cost <= eps):
                # Восстанавливаем полное решение
                x = np.zeros(n_total)
                x[basis] = x_B
                # Возвращаем только исходные переменные (без слаков)
                return "optimal", x[:n], np.dot(c[:n], x[:n])

            # 6. Выбор входящей переменной (правило наибольшей стоимости)
            entering_pos = np.argmax(reduced_cost)
            entering_index = nbasis[entering_pos]

            # 7. Вычисляем направление d = B^{-1} * a_entering
            a_entering = A[:, entering_index]
            d = np.linalg.solve(B, a_entering)

            # 8. Проверка на неограниченность
            if np.all(d <= eps):
                return "unbounded", None, None

            # 9. Выбор выходящей переменной (правило минимального отношения)
            ratios = []
            for i in range(m):
                if d[i] > eps:
                    ratios.append(x_B[i] / d[i])
                else:
                    ratios.append(np.inf)

            leaving_pos = np.argmin(ratios)
            leaving_index = basis[leaving_pos]

            # 10. Обновляем базисы
            basis[leaving_pos] = entering_index
            nbasis[entering_pos] = leaving_index

        except np.linalg.LinAlgError:
            return "error", None, None

    return "max_iterations_reached", None, None


def Phase1(c, A, b):
    m, n_total = A.shape
    n = n_total - m  # исходные переменные

    # Проверяем, есть ли отрицательные правые части
    if np.all(b >= -eps):
        # Если все b >= 0, можем сразу запустить Фазу II
        return PrimalSimplex(c, A, b)

    # Создаем вспомогательную задачу
    # Добавляем одну искусственную переменную x0
    new_c = np.zeros(n_total + 1)
    new_c[-1] = -1  # минимизируем -x0 (эквивалентно максимизации x0)

    # Расширяем матрицу A: Ax + s - x0 * 1 = b
    new_A = np.zeros((m, n_total + 1))
    new_A[:, :n_total] = A
    new_A[:, n_total] = -1  # коэффициент при x0

    # Начальный базис: слаки + x0
    # Выбираем строку с наиболее отрицательным b
    most_negative_idx = np.argmin(b)
    basis = list(range(n, n_total))  # слаки
    basis[most_negative_idx] = n_total  # заменяем один слак на x0
    nbasis = list(range(0, n))  # исходные переменные

    # Решаем вспомогательную задачу
    status, x, obj = PrimalSimplex(new_c, new_A, b, basis, nbasis)

    # Проверяем результат Фазы I
    if status != "optimal" or obj > eps:
        return "unbounded", None, None

    # Если x0 > 0 в решении, задача недопустима
    if x[-1] > eps:
        return "unbounded", None, None

    # Восстанавливаем исходную задачу и запускаем Фазу II
    return PrimalSimplex(c, A, b)


def Solve(c, A, b):
    m, n_orig = A.shape

    # Преобразуем задачу в стандартную форму: Ax + s = b, s >= 0
    # Добавляем слаковые переменные
    n_total = n_orig + m
    new_A = np.zeros((m, n_total))
    new_A[:, :n_orig] = A
    new_A[:, n_orig:] = np.eye(m)  # матрица для слаков

    new_c = np.zeros(n_total)
    new_c[:n_orig] = c

    # Проверяем допустимость начального решения
    if np.all(b >= -eps):
        return PrimalSimplex(new_c, new_A, b)
    else:
        return Phase1(new_c, new_A, b)


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
