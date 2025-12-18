import numpy as np
import sys
import argparse

from constants import *
from factorized_matrix import FactorizedMatrix


def inf_norm(x):
    return np.abs(x.flatten()).max()

def get_maximum_allowed_step(x, dx):
    x = x.flatten()
    dx = dx.flatten()
    if not (dx < -EPS).any():
        return np.inf

    mask = dx < -EPS
    return (- x[mask] / dx[mask]).min()


class InternalPointState:
    def __init__(self, c, A, b):
        m, n = A.shape
        self.c, self.A, self.b = c, A, b

        self.em, self.en = np.ones((m, 1)), np.ones((n, 1))
        self.x = self.en.copy()
        self.z = self.en.copy()
        self.y = self.em.copy()
        self.w = self.em.copy()

        self.m, self.n = m, n

    def is_optimal(self, sigma, rho, mu_hat) -> bool:
        is_rho_optimal = inf_norm(rho) < EPS_RHO * (1 + inf_norm(self.b))
        is_sigma_optimal = inf_norm(sigma) < EPS_SIGMA * (1 + inf_norm(self.c))
        is_mu_hat_optimal = mu_hat < EPS_MU
        return is_rho_optimal and is_sigma_optimal and is_mu_hat_optimal

    def is_positive(self) -> bool:
        for v in (self.x, self.z, self.y, self.w):
            if not (v > EPS).all():
                return False
        return True

    def is_bounded(self) -> bool:
        for v in (self.x, self.z, self.y, self.w):
            if inf_norm(v) > UPPER_BOUND:
                return False
        return True

    def compute_linearization_matrix(self) -> FactorizedMatrix:
        n, m = self.n, self.m

        M = np.zeros((n + m, n + m))
        M[ :n,  :n] =   np.diagflat(self.z / self.x)
        M[ :n, n: ] = self.A.T
        M[n: ,  :n] = self.A
        M[n: , n: ] = - np.diagflat(self.w / self.y)
        return FactorizedMatrix(M)

    def compute_objective(self):
        return self.x.T @ self.c

    def compute_constraints_compatibility(self):
        rho   = self.b - self.A @ self.x - self.w
        sigma = self.c - self.A.T @ self.y + self.z
        return sigma, rho

    def compute_dual_gap(self):
        gamma = (self.x * self.z).sum() + (self.y * self.w).sum()
        return gamma

    def compute_complementarity_rhs(self, sigma, rho, mu, xz_affine_correction=0.0, yw_affine_correction=0.0):
        r_xz = mu * self.en - self.x * self.z + xz_affine_correction
        r_yw = mu * self.em - self.y * self.w + yw_affine_correction

        rhs = np.concatenate([
            sigma + r_xz / self.x,
            rho   - r_yw / self.y
        ], axis=0)
        return rhs

    def compute_delta_split(self, s, sigma, rho):
        dx = s[:self.n, :]
        dy = s[self.n:, :]
        dz = self.A.T @ dy - sigma
        dw = rho - self.A @ dx
        return dx, dy, dz, dw

    def compute_maximum_allowed_step(self, dx, dy, dz, dw):
        updates = ((self.x, dx), (self.y, dy), (self.z, dz), (self.w, dw))
        max_allowed_steps = [
            get_maximum_allowed_step(v, dv)
            for v, dv in updates
        ]
        return min(max_allowed_steps)

    def update(self, dx, dy, dz, dw, theta):
        self.x += theta * dx
        self.y += theta * dy
        self.z += theta * dz
        self.w += theta * dw

    def copy(self):
        state = InternalPointState(self.c, self.A, self.b)
        state.x = self.x.copy()
        state.z = self.z.copy()
        state.w = self.w.copy()
        state.y = self.y.copy()
        return state


def solve(c, A, b):
    m, n = A.shape
    n -= m

    c = c.reshape((n + m, 1))[:n, :]
    A = A[:, :n]
    b = b.reshape((m, 1))

    state = InternalPointState(c, A, b)

    for step in range(MAX_ITER):
        # 1. Невязки допустимости.
        #    Оценка того, насколько далеко текущее решение от удовлетворения ограничениям.
        sigma, rho = state.compute_constraints_compatibility()

        # 2. Duality gap.
        #    Вычисление комплементарного зазора.
        gamma = state.compute_dual_gap()
        mu_hat = gamma / (m + n)

        # 3. Проверка оптимальности.
        #    Если невязки допустимости и зазор очень малы, то считаем что сошлись.
        if state.is_optimal(sigma, rho, mu_hat):
            return "optimal", state.x, state.compute_objective()

        # 4. Матрица системы линеаризации и ее факторизация.
        #    Инициализируем редуцированную матрицу (10) и вычисляем ее факторизацию.
        M: FactorizedMatrix = state.compute_linearization_matrix()
        M.init_factorization()

        # 5. Прогноз корректора.
        #    Вычисляем направление центрального пути.

        #    Копируем текущее состояние и делаем шаг.
        #    В C++ можно было бы не делать лишних копирований с помощью shared_ptr.
        predicted_state = state.copy()

        rhs_affine = predicted_state.compute_complementarity_rhs(sigma, rho, mu=0.0)
        assert rhs_affine.shape == (n + m, 1)

        #    Решаем систему через предпосчитанную LU-декомпозицию.
        delta_affine = M.solve(rhs_affine)
        dx_affine, dy_affine, dz_affine, dw_affine = predicted_state.compute_delta_split(delta_affine, sigma, rho)

        #    Вычисляем максимальный разрешимый шаг корректора.
        max_allowed_affine_step = predicted_state.compute_maximum_allowed_step(dx_affine, dy_affine, dz_affine, dw_affine)
        theta_affine_max = min(1.0, max_allowed_affine_step)
        theta_affine = ALPHA_AFFINE * theta_affine_max

        predicted_state.update(dx_affine, dy_affine, dz_affine, dw_affine, theta_affine)

        predicted_gamma = predicted_state.compute_dual_gap()
        predicted_mu_hat = predicted_gamma / (n + m)

        correction = np.power(predicted_mu_hat / mu_hat, PREDICTION_POWER)

        # 6. Определение внутренней точки.
        #    Вычислив коррекцию по направлению центрального пути, корректируем внутреннюю точку.
        mu_hat = correction * mu_hat
        mu = DELTA * mu_hat

        # 7. Шаг корректора.
        #    Вычисляем шаг по центральному пути.
        xz_affine_correction = - dx_affine * dz_affine
        yw_affine_correction = - dy_affine * dw_affine
        rhs = state.compute_complementarity_rhs(sigma, rho, mu, xz_affine_correction, yw_affine_correction)
        assert rhs.shape == (n + m, 1)

        delta = M.solve(rhs)
        dx, dy, dz, dw = state.compute_delta_split(delta, sigma, rho)

        max_allowed_step = state.compute_maximum_allowed_step(dx, dy, dz, dw)
        theta_max = min(1.0, max_allowed_step)
        theta = ALPHA * theta_max

        state.update(dx, dy, dz, dw, theta)

        # 8. Проверка на положительность и ограниченность.
        #    Если алгоритм достигает EPS,
        #    то это значит, что rho и sigma (невязки допустимости) все еще велики,
        #    и ограничением шага является только положительность решения. Тогда
        #    наша задача с какой-то стороны не ограничена.
        if not state.is_positive() or not state.is_bounded():
            return "unbounded", None, None

    return "max_iter", state.x, state.compute_objective()


def proc_cmd():
    parser = argparse.ArgumentParser(description="Solve a linear program using the Internal Point method.")
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

    A_full = np.hstack([A, np.eye(m)])
    c_full = np.concatenate([c, np.zeros(m)])

    print("n =", n)
    print("m =", m)
    print("c =", c)
    print("A =", A)
    print("b =", b)

    print("Solving the linear program using the Internal Point method...\n")
    status, solution, objective = solve(c_full, A_full, b)
    print("\nResult:")
    print("Status:", status)
    if status == "optimal":
        print("Optimal solution x* =", solution)
        print("Optimal value =", objective)
    elif status == "unbounded":
        print("The problem is unbounded.")
    elif status == "max_iter":
        print("Algorithm did not converge.")
        print("Found solution x* =", solution)
        print("Found value =", objective)
    else:
        print("No solution found.")


if __name__ == '__main__':
    main()
