
- Выбранный алгоритм: (2) Mehrotra Predictor-Corrector.
- Система для решения: (10) Редуцированная симметричная система.
- Гиперпараметры алгоритма: в файле `constants.py`.
- Где переиспользуется факторизация: редуцированная симметрическая система
имеет матрицу `M`, которая представлена классом `FactorizedMatrix` (реализация в 
файле `factorized_matrix.py`). На каждой итерации метода, один раз строится `M`
и один раз вычисляется LU факторизация - `M.init_factorization()` - и
решаются две системы `M.solve(rhs_affine)` и `M.solve(rhs)` без пересчета факторизации.

В решении не использовалась разреженность.

Наблюдения: метод Мехротры сходится чуть лучше. Например, на `example_stalling.txt`.


<details><summary>Вывод программы.</summary><p>

<pre>
$ python ipm_mehrotra.py example_phase1.txt
n = 2
m = 3
c = [2. 1.]
A = [[-1.  1.]
 [-1. -2.]
 [ 0.  1.]]
b = [-1. -2.  1.]
Solving the linear program using the Internal Point method...


Result:
Status: unbounded
The problem is unbounded.

$ python ipm_mehrotra.py example_phase3.txt
n = 3
m = 4
c = [3. 2. 1.]
A = [[1. 1. 1.]
 [2. 1. 0.]
 [1. 3. 1.]
 [0. 0. 1.]]
b = [30. 40. 60. 10.]
Solving the linear program using the Internal Point method...


Result:
Status: optimal
Optimal solution x* = [[12.71105519]
 [14.57788963]
 [ 2.71105519]]
Optimal value = [[70.]]

$ python ipm_mehrotra.py example_stalling.txt 
n = 4
m = 3
c = [ 1. -2.  0. -2.]
A = [[ 0.5 -3.5 -2.   4. ]
 [ 0.5  0.  -0.5  0.5]
 [ 1.   0.   0.   0. ]]
b = [0. 0. 1.]
Solving the linear program using the Internal Point method...


Result:
Status: optimal
Optimal solution x* = [[1.00000000e+00]
 [3.98882999e-09]
 [1.06215574e+00]
 [1.52107931e-09]]
Optimal value = [[0.99999999]]

$ python ipm_mehrotra.py example_exp_iters.txt 
n = 7
m = 7
c = [64. 32. 16.  8.  4.  2.  1.]
A = [[  1.   0.   0.   0.   0.   0.   0.]
 [  4.   1.   0.   0.   0.   0.   0.]
 [  8.   4.   1.   0.   0.   0.   0.]
 [ 16.   8.   4.   1.   0.   0.   0.]
 [ 32.  16.   8.   4.   1.   0.   0.]
 [ 64.  32.  16.   8.   4.   1.   0.]
 [128.  64.  32.  16.   8.   4.   1.]]
b = [1.e+00 1.e+02 1.e+04 1.e+06 1.e+08 1.e+10 1.e+12]
Solving the linear program using the Internal Point method...


Result:
Status: optimal
Optimal solution x* = [[5.61380182e-12]
 [3.94782439e-12]
 [2.19788881e-11]
 [3.06034022e-11]
 [7.36640670e-11]
 [1.83158866e-10]
 [1.00000000e+12]]
Optimal value = [[1.e+12]]
</pre>

</p></details>
