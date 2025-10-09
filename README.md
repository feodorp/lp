# Домашнее задание: Фаза 1, Фаза 2, Прямой Симплекс Метод

- Реализовал обе фазы simplex метода, используя предложенный шаблон, используя LU факторизацию матрицы B, которая обновляется по мере необходимости. 

- Для выбора следующей переменной использую правило Бланда, выбирая наименьшую по индексу переменную, увеличивающую целевую функцию. 

- В качестве численного допуска для всех алгоритмов использую eps=1e-8
- LU декомпозицию считаю 1 раз вначале запуска на единичной матрице B, затем использую элиминацию гаусса, чтобы решать уравнения вида $Bx=y \iff LUx=y$ и в конце каждой итерации пытаюсь обновить колонку в матрице, в случае неудачи пересчитываю факторизацию.


Локальный вывод:

<details><summary>Вывод программы</summary>

<pre>
$ python simplex_template --filename example_phase1.txt
n = 2
m = 3
c = [2. 1.]
A = [[-1.  1.]
 [-1. -2.]
 [ 0.  1.]]
b = [-1. -2.  1.]
Solving the linear program using the Primal Simplex method...


Result:
Status: unbounded
The problem is unbounded.

$ python simplex_template --filename example_phase2.txt
n = 3
m = 4
c = [3. 2. 1.]
A = [[1. 1. 1.]
 [2. 1. 0.]
 [1. 3. 1.]
 [0. 0. 1.]]
b = [30. 40. 60. 10.]
Solving the linear program using the Primal Simplex method...


Result:
Status: optimal
Optimal solution x* = [12.5 15.   2.5]
Optimal value = 70.0

</pre>

</details>