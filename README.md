# Домашнее задание: Фаза 1, Фаза 2, Прямой Симплекс Метод
* Реализовал Фазу 1 и Фазу 2 для симплекс метода

* Для обновения базиса использовал LU-разложение, при нулевом пивоте добавлял $\varepsilon = 10^{-9}$ и пересчитывал

## Дизайн решения:
### 1. Структура данных
Решали задачу:
$$
\begin{array}{ll}
\text{максимизировать} & c^{T}x \\[6pt]
\text{при ограничениях} & A x \leq b, \\[4pt]
& x \geq 0.
\end{array}
$$
Формат данных:
* Первая строка: $n$ (количество переменных) и $m$ (количество ограничений).
* Вторая строка: вектор целевой функции c (длиной $n$).
* Следующие $m$ строк: каждая строка матрицы $A$, за которой следует соответствующий элемент $b_i$.

### 2. Правило разрешения ничьих
Правило Блэнда: брал переменную с наименьшим индексом из подходящих кандидатов.

### 3. Численные допуски
Использовал $eps = 10^{-9}$.

## Лог запуска:
<details><summary>Вывод программы</summary><p>

<pre>
$ python simplex_template.py example_phase1.txt

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

$ python simplex_template.py example_phase2.txt

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

</p></details>
