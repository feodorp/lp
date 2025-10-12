# Домашнее задание: Фаза 1, Фаза 2, Прямой Симплекс Метод

## Архитектура
 - Использовано обновление LU разложения
 - Используется правило Блэнда
 - Допуск равен 1e-9
 - Функции, связанные с LU находятся в файле lu.py

## Проделанная работа:
### 1. Реализован простой симплекс-метод
Описан функцией 
```python
def PrimalSimplex_without_LU(c, A, b, basis=None, nbasis=None):
    ...
```
в конце файла

### 3. Реализовано LU-разложение
Описан функцией 
```python
def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    ...
```
в начале файла


## Вывод программы на питоне

<details><summary>Вывод программы</summary><p>

<pre>
$ python simplex_template --filename example_phase1.txt
n = 2
m = 3
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
Optimal solution x* = [ 12.  16.   2.   0.   0.   0. -30.]
Optimal value = 70.0
</pre>

</p></details>

---