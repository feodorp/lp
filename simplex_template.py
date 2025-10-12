#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import argparse

EPS = 1e-9


# ===================================================
#            LU-РАЗЛОЖЕНИЕ И РЕШЕНИЕ СЛАУ
# ===================================================

def lu_decompose(A: np.ndarray):
    A = np.array(A, dtype=float, copy=True)
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()
    for k in range(n):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if abs(U[pivot, k]) < 1e-14:
            raise np.linalg.LinAlgError("LU: нулевой пивот")
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
        L[k, k] = 1.0
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    return P, L, U


def lu_solve(P: np.ndarray, L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    pb = P @ b
    y = np.zeros_like(pb)
    n = L.shape[0]
    for i in range(n):
        y[i] = pb[i] - L[i, :i] @ y[:i]
    x = np.zeros_like(y)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < 1e-14:
            raise np.linalg.LinAlgError("LU-solve: нулевой пивот")
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def invert_via_lu(B: np.ndarray) -> np.ndarray:
    P, L, U = lu_decompose(B)
    m = B.shape[0]
    I = np.eye(m)
    cols = [lu_solve(P, L, U, I[:, i]) for i in range(m)]
    return np.column_stack(cols)


# ===================================================
#          ОБНОВЛЕНИЕ МАТРИЦЫ ПО ФОРМУЛЕ
#               ШЕРМАНА–МОРРИСОНА
# ===================================================

def sherman_morrison_update(B_inv: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = u.reshape(-1)
    v = v.reshape(-1)
    Binv_u = B_inv @ u
    vT_Binv = v @ B_inv
    denom = 1.0 + vT_Binv @ u
    if abs(denom) < 1e-14:
        raise np.linalg.LinAlgError("Sherman–Morrison: denominator ≈ 0")
    return B_inv - np.outer(Binv_u, vT_Binv) / denom


# ===================================================
#             РЕВИЗОВАННЫЙ СИМПЛЕКС-МЕТОД
# ===================================================

def revised_simplex(A: np.ndarray, b: np.ndarray, c: np.ndarray,
                    B_idx=None, N_idx=None):
    m, total = A.shape
    n = total - m
    if B_idx is None:
        B_idx = list(range(n, n + m))
    if N_idx is None:
        N_idx = [j for j in range(total) if j not in B_idx]
    B = A[:, B_idx]
    try:
        B_inv = invert_via_lu(B)
    except np.linalg.LinAlgError:
        return "infeasible", None, None
    it_lim = 20000
    for _ in range(it_lim):
        b_bar = B_inv @ b
        c_B = c[B_idx]
        y = B_inv.T @ c_B
        r = np.zeros(total)
        if N_idx:
            A_N = A[:, N_idx]
            r_N = c[N_idx] - A_N.T @ y
            r[N_idx] = r_N
        entering_candidates = [j for j in N_idx if r[j] > EPS]
        if not entering_candidates:
            x_full = np.zeros(total)
            for pos, j in enumerate(B_idx):
                x_full[j] = b_bar[pos]
            return "optimal", x_full[:n], float(c @ x_full)
        j_enter = min(entering_candidates)
        a_j = A[:, j_enter]
        d = B_inv @ a_j
        if np.all(d <= EPS):
            return "unbounded", None, None
        theta = np.inf
        p_row = -1
        tie_indices = []
        for i in range(m):
            if d[i] > EPS:
                val = b_bar[i] / d[i]
                if val < theta - EPS:
                    theta = val
                    p_row = i
                    tie_indices = [i]
                elif abs(val - theta) <= EPS:
                    tie_indices.append(i)
        if p_row < 0 or not np.isfinite(theta):
            return "unbounded", None, None
        if len(tie_indices) > 1:
            p_row = min(tie_indices, key=lambda i: B_idx[i])
        j_leave = B_idx[p_row]
        b_p = A[:, j_leave]
        u = a_j - b_p
        e_p = np.zeros(m)
        e_p[p_row] = 1.0
        try:
            B_inv = sherman_morrison_update(B_inv, u, e_p)
        except np.linalg.LinAlgError:
            B = A[:, B_idx].copy()
            B[:, p_row] = a_j
            try:
                B_inv = invert_via_lu(B)
            except np.linalg.LinAlgError:
                return "infeasible", None, None
        B_idx[p_row] = j_enter
        N_idx = [j for j in range(total) if j not in B_idx]
    return "infeasible", None, None


# ===================================================
#                    ФАЗА I
# ===================================================

def phase1(c: np.ndarray, A: np.ndarray, b: np.ndarray):
    m, n = A.shape
    I = np.eye(m)
    A_full = np.hstack([A, I, I])
    c_full = np.hstack([np.zeros(n + m), -np.ones(m)])
    B_idx = list(range(n + m, n + m + m))
    N_idx = [j for j in range(n + m + m) if j not in B_idx]
    status_I, x_full, obj_I = revised_simplex(A_full, b, c_full, B_idx, N_idx)
    PHASE1_EPS = 1e-7
    if status_I != "optimal" or obj_I > PHASE1_EPS:
        return "infeasible", None, None
    A_no_a = A_full[:, :n + m]
    B_idx2 = [j for j in B_idx if j < n + m]
    if not B_idx2:
        B_idx2 = list(range(n, n + m))
    N_idx2 = [j for j in range(n + m) if j not in B_idx2]
    c_ext = np.hstack([c, np.zeros(m)])
    return revised_simplex(A_no_a, b, c_ext, B_idx2, N_idx2)


# ===================================================
#                    ФАЗА II
# ===================================================

def Solve(c: np.ndarray, A: np.ndarray, b: np.ndarray):
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape
    if np.all(b >= -EPS):
        I = np.eye(m)
        A_ext = np.hstack([A, I])
        c_ext = np.hstack([c, np.zeros(m)])
        return revised_simplex(A_ext, b, c_ext)
    else:
        return phase1(c, A, b)


# ===================================================
#                ОСНОВНОЙ ВЫЗОВ (CLI)
# ===================================================

def main():
    parser = argparse.ArgumentParser(description="Решение задачи ЛП симплекс-методом (Phase I/II, Sherman–Morrison + LU).")
    parser.add_argument("filename", type=str, help="Имя входного файла с задачей ЛП.")
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
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
    print("========== ВХОДНЫЕ ДАННЫЕ ==========")
    print(f"n = {n}")
    print(f"m = {m}")
    print(f"c = {c}")
    print("A =")
    print(A)
    print(f"b = {b}")
    print("====================================\n")
    print("Решение задачи линейного программирования...\n")
    status, x, obj = Solve(c, A, b)
    print("============= РЕЗУЛЬТАТ =============")
    print("Статус:", status)
    if status == "optimal":
        print("Оптимальное решение x* =", x)
        print("Оптимальное значение =", obj)
    elif status == "unbounded":
        print("Целевая функция не ограничена (unbounded).")
    else:
        print("Система несовместна (infeasible).")
    print("====================================\n")


if __name__ == "__main__":
    main()
