import numpy as np
import argparse

eps = 1e-9



# =========================================
#                Фаза I
# =========================================
def Phase1(c, A, b):
    m, n = A.shape
    I = np.eye(m)
    A_ext = np.hstack([A, I])
    a0 = -np.ones((m, 1))
    A_aux = np.hstack([A_ext, a0])
    c_aux = np.hstack([np.zeros(n + m), -1.0])
    p = int(np.argmin(b))
    x0_col = n + m
    basis = list(range(n, n + m))
    basis[p] = x0_col
    nbasis = [j for j in range(n + m + 1) if j not in basis]
    status, _, obj = PrimalSimplex(c_aux, A_aux, b, basis, nbasis)
    if status != "optimal" or obj > eps:
        return "infeasible", None, None
    A_no_x0 = A_aux[:, :n + m]
    basis_wo = []
    for idx in basis:
        if idx == x0_col:
            basis_wo.append(n)
        elif idx < x0_col:
            basis_wo.append(idx)
        else:
            basis_wo.append(idx - 1)
    nbasis_wo = [j for j in range(n + m) if j not in basis_wo]
    c_ext = np.hstack([c, np.zeros(m)])
    return PrimalSimplex(c_ext, A_no_x0, b, basis_wo, nbasis_wo)

# =========================================
#   Прямой симплекс-метод (Фаза II)
# =========================================
def PrimalSimplex(c, A, b, basis=None, nbasis=None):
    m, total = A.shape
    n = total - m
    if basis is None or nbasis is None:
        basis  = list(range(n, n + m))
        nbasis = list(range(0, n))

    iter_lim = 10000
    for _ in range(iter_lim):
        B = A[:, basis]
        N = A[:, nbasis]
        c_B = c[basis]
        try:
            y = np.linalg.solve(B.T, c_B)
        except np.linalg.LinAlgError:
            return "infeasible", None, None
        r_N = c[nbasis] - N.T @ y
        pos = np.where(r_N > eps)[0]
        if pos.size == 0:
            try:
                x_B = np.linalg.solve(B, b)
            except np.linalg.LinAlgError:
                return "infeasible", None, None
            x_full = np.zeros(total)
            x_full[basis] = x_B
            return "optimal", x_full[:n], float(c @ x_full)
        entering_pos = int(pos[0])
        entering_index = nbasis[entering_pos]
        a_j = A[:, entering_index]
        try:
            d = np.linalg.solve(B, a_j)
        except np.linalg.LinAlgError:
            return "infeasible", None, None
        if np.all(d <= eps):
            return "unbounded", None, None
        try:
            x_B = np.linalg.solve(B, b)
        except np.linalg.LinAlgError:
            return "infeasible", None, None
        ratios = np.full(m, np.inf)
        mask = d > eps
        ratios[mask] = x_B[mask] / d[mask]
        p_row = int(np.argmin(ratios))
        if not np.isfinite(ratios[p_row]):
            return "unbounded", None, None
        leaving_index = basis[p_row]
        basis[p_row] = entering_index
        nbasis[entering_pos] = leaving_index
    return "infeasible", None, None


# =========================================
#         LU-разложение и решение
# =========================================
def lu_decompose(A):
    A = np.array(A, float)
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()
    for k in range(n):
        pivot = np.argmax(np.abs(U[k:, k])) + k
        if abs(U[pivot, k]) < 1e-14:
            raise np.linalg.LinAlgError("LU: zero pivot")
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

def lu_solve(P, L, U, b):
    b = np.asarray(b, float)
    pb = P @ b
    y = np.zeros_like(pb)
    n = L.shape[0]
    for i in range(n):
        y[i] = pb[i] - L[i, :i] @ y[:i]
    x = np.zeros_like(y)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x

def invert_via_lu(B):
    P, L, U = lu_decompose(B)
    m = B.shape[0]
    I = np.eye(m)
    cols = [lu_solve(P, L, U, I[:, i]) for i in range(m)]
    return np.column_stack(cols)


# =========================================
#     Шерман–Моррисон обновление
# =========================================
def sherman_morrison_update(B_inv, u, v):
    u = u.reshape(-1)
    v = v.reshape(-1)
    Binv_u = B_inv @ u
    vT_Binv = v @ B_inv
    denom = 1.0 + vT_Binv @ u
    if abs(denom) < 1e-14:
        raise np.linalg.LinAlgError("Sherman–Morrison: denom≈0")
    return B_inv - np.outer(Binv_u, vT_Binv) / denom


# =========================================
#      Ревизованный симплекс (SM + LU)
# =========================================
def RevisedSimplex_SM(A, b, c, B_idx=None, N_idx=None):
    m, total = A.shape
    n = total - m
    if B_idx is None:
        B_idx = list(range(n, n + m))
    if N_idx is None:
        N_idx = [j for j in range(total) if j not in B_idx]
    try:
        B_inv = invert_via_lu(A[:, B_idx])
    except np.linalg.LinAlgError:
        return "infeasible", None, None

    for _ in range(20000):
        b_bar = B_inv @ b
        c_B = c[B_idx]
        y = B_inv.T @ c_B
        r = np.zeros(total)
        if N_idx:
            r[N_idx] = c[N_idx] - A[:, N_idx].T @ y
        entering = [j for j in N_idx if r[j] > eps]
        if not entering:
            x_full = np.zeros(total)
            for pos, j in enumerate(B_idx):
                x_full[j] = b_bar[pos]
            return "optimal", x_full[:n], float(c @ x_full)
        j_enter = min(entering)
        a_j = A[:, j_enter]
        d = B_inv @ a_j
        if np.all(d <= eps):
            return "unbounded", None, None
        theta = np.inf
        p_row = -1
        for i in range(m):
            if d[i] > eps:
                val = b_bar[i] / d[i]
                if val < theta - eps:
                    theta = val
                    p_row = i
        if p_row < 0 or not np.isfinite(theta):
            return "unbounded", None, None
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
            B_inv = invert_via_lu(B)
        B_idx[p_row] = j_enter
        N_idx = [j for j in range(total) if j not in B_idx]
    return "infeasible", None, None


# =========================================
#         Фаза I (с SM и LU)
# =========================================
def Phase1_SM(c, A, b):
    m, n = A.shape
    I = np.eye(m)
    A_ext = np.hstack([A, I])
    a0 = -np.ones((m, 1))
    A_aux = np.hstack([A_ext, a0])
    c_aux = np.hstack([np.zeros(n + m), -1.0])
    p = int(np.argmin(b))
    x0_col = n + m
    B_idx = list(range(n, n + m))
    B_idx[p] = x0_col
    N_idx = [j for j in range(n + m + 1) if j not in B_idx]
    status, _, obj = RevisedSimplex_SM(A_aux, b, c_aux, B_idx, N_idx)
    if status != "optimal" or obj > eps:
        return "infeasible", None, None
    A_no_x0 = A_aux[:, :n + m]
    B_idx2 = []
    for j in B_idx:
        if j == x0_col:
            B_idx2.append(n)
        elif j < x0_col:
            B_idx2.append(j)
        else:
            B_idx2.append(j - 1)
    N_idx2 = [j for j in range(n + m) if j not in B_idx2]
    c_ext = np.hstack([c, np.zeros(m)])
    return RevisedSimplex_SM(A_no_x0, b, c_ext, B_idx2, N_idx2)


# =========================================
#            Обёртка для запуска
# =========================================
def Solve_SM(c, A, b):
    c = np.asarray(c, float)
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    if np.all(b >= -eps):
        m, n = A.shape
        I = np.eye(m)
        A_ext = np.hstack([A, I])
        c_ext = np.hstack([c, np.zeros(m)])
        return RevisedSimplex_SM(A_ext, b, c_ext)
    else:
        return Phase1_SM(c, A, b)


# =========================================
#        Интерфейсы запуска
# =========================================
def run_from_file(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        n, m = map(int, f.readline().split())
        c = np.array(list(map(float, f.readline().split())))
        A = []
        b = []
        for _ in range(m):
            *row, bi = map(float, f.readline().split())
            A.append(row)
            b.append(bi)
    A = np.array(A, float)
    b = np.array(b, float)
    print("n =", n)
    print("m =", m)
    print("c =", c)
    print("A =\n", A)
    print("b =", b)
    print("\nSolving...\n")
    status, x, obj = Solve_SM(c, A, b)
    print("Result:")
    print("Status:", status)
    if status == "optimal":
        print("x* =", x)
        print("objective =", obj)
    elif status == "unbounded":
        print("The problem is unbounded.")
    else:
        print("No feasible solution (infeasible).")

def run_from_arrays(c, A, b):
    status, x, obj = Solve_SM(c, A, b)
    print("Status:", status)
    if status == "optimal":
        print("x* =", x)
        print("objective =", obj)
    elif status == "unbounded":
        print("The problem is unbounded.")
    else:
        print("Infeasible.")
    return status, x, obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplex solver (Phase I/II + LU + SM).")
    parser.add_argument("filename", type=str, help="Input file with LP.")
    args = parser.parse_args()
    run_from_file(args.filename)
