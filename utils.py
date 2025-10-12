import numpy as np
from typing import List

from constants import EPS


def pivot_operation(basis: List[int], nbasis: List[int], leaving_index: int, entering_index: int):
    assert leaving_index in basis and entering_index not in basis
    assert entering_index in nbasis and leaving_index not in nbasis

    len_basis = len(basis)
    len_nbasis = len(nbasis)

    basis, nbasis = np.array(basis), np.array(nbasis)
    basis[basis == leaving_index] = entering_index
    nbasis[nbasis == entering_index] = leaving_index
    # basis  = [entering_index] + [i for i in basis if i  != leaving_index]
    # nbasis = [leaving_index]  + [j for j in nbasis if j != entering_index]

    assert len_basis == len(basis) and len_nbasis == len(nbasis)
    return basis.tolist(), nbasis.tolist()

def forward_substitution(L, b):
    assert L.shape[0] == L.shape[1]

    n = L.shape[0]
    x = np.zeros(n)
    for r in range(n):
        dot_product = L[r, :r] @ x[:r]
        x[r] = (b[r] - dot_product) / L[r, r]
    return x

def backward_substitution(U, b):
    assert U.shape[0] == U.shape[1]

    n = U.shape[0]
    x = np.zeros(n)
    for r in range(n - 1, -1, -1):
        dot_product = U[r, r:] @ x[r:]
        x[r] = (b[r] - dot_product) / U[r, r]
    return x

def solve(P, L, U, b):
    # solve Bx = b for x, where B = P @ L @ U
    # Bx = b -> x = B_inv @ b
    # B_inv = U_inv @ L_inv @ P_inv
    # 1) solve Pv = b for v ---> v = P_inv @ b
    # 2) solve Ly = v for y ---> y = L_inv @ v =         L_inv @ P_inv @ b
    # 3) solve Uz = y for z ---> z = U_inv @ y = U_inv @ L_inv @ P_inv @ b
    # P_inv = P.T since permutation matrix is orthogonal
    v = P.T @ b
    y = forward_substitution(L, v)
    z = backward_substitution(U, y)
    return z

def solve_transpose(P, L, U, b):
    # solve B.T x = b for x, where B = P @ L @ U, B.T = U.T @ L.T @ P.T
    # B.T x = b -> x = B.T_inv @ b
    # B.T_inv = P.T_inv @ L.T_inv @ U.T_inv
    # 1) solve U.T v = b for v ---> v = U.T_inv @ b
    # 2) solve L.T y = v for y ---> y = L.T_inv @ v =           L.T_inv @ U.T_inv @ b
    # 3) solve P.T z = y for z ---> z = P.T_inv @ y = P.T_inv @ L.T_inv @ U.T_inv @ b
    v = forward_substitution(U.T, b)
    y = backward_substitution(L.T, v)
    z = P @ y
    return z

def update(P, L, U, entering_column, l):
    m = L.shape[0]

    # v = (PL)^{-1} @ entering_column
    v = forward_substitution(L, P.T @ entering_column)
    H = U.copy()
    H[:, l] = v

    G_inv = np.eye(m)
    for c in range(l, m - 1):
        if abs(H[c, c]) < EPS:
            raise ValueError(f"Small pivot at position ({c}, {c})")

        G_k_inv = np.eye(m)
        for k in range(c + 1, m):
            if abs(H[k, c]) < EPS:
                continue
            mu = H[k, c] / H[c, c]
            G_k_inv[k, c] = mu
            H[k, c:] -= mu * H[c, c:]

        G_inv = G_inv @ G_k_inv

    U_new = H
    L_new = L.copy() @ G_inv
    # print(f"U_new=\n{U_new}")
    # print(f"L_new=\n{L_new}")
    return L_new, U_new
