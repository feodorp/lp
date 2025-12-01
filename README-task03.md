Я реализовал дуальный симплекс метод без фазы 1. В качестве обновления
обратной матрицы использую формулу Шермана-Моррисона, также как в первом дз.

В качестве верхней границы взял $1e20$.

<details><summary>Вывод программы (simplex_dual.py)</summary><p>

<pre>
$ python simplex_dual.py example_phase1.txt
n = 2
m = 3
c = [2. 1.]
A = [[-1.  1.]
 [-1. -2.]
 [ 0.  1.]]
b = [-1. -2.  1.]
Solving the linear program using the Dual Simplex method...


Result:
Status: unbounded
The problem is unbounded.

$ python simplex_dual.py example_phase2.txt
n = 3
m = 4
c = [3. 2. 1.]
A = [[1. 1. 1.]
 [2. 1. 0.]
 [1. 3. 1.]
 [0. 0. 1.]]
b = [30. 40. 60. 10.]
Solving the linear program using the Dual Simplex method...


Result:
Status: optimal
Optimal solution x* = [12.5 15.   2.5]
Optimal value = 70.0

$ python simplex_dual.py example_stalling.txt
n = 4
m = 3
c = [ 1. -2.  0. -2.]
A = [[ 0.5 -3.5 -2.   4. ]
 [ 0.5  0.  -0.5  0.5]
 [ 1.   0.   0.   0. ]]
b = [0. 0. 1.]
Solving the linear program using the Dual Simplex method...


Result:
Status: optimal
Optimal solution x* = [1. 0. 1. 0.]
Optimal value = 1.0

$ python simplex_dual.py example_exp_iters.txt
n = 4
m = 4
c = [8. 4. 2. 1.]
A = [[ 1.  0.  0.  0.]
 [ 4.  1.  0.  0.]
 [ 8.  4.  1.  0.]
 [16.  8.  4.  1.]]
b = [1.e+00 1.e+02 1.e+04 1.e+06]
Solving the linear program using the Dual Simplex method...


Result:
Status: optimal
Optimal solution x* = [      0.       0.       0. 1000000.]
Optimal value = 1000000.0

</pre>

</p></details>