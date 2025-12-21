import numpy as np
import argparse

# help functions
def get_theta(v, dv):
    mask = dv < 0
    if np.any(mask):
        return min(1.0, float(np.min(-v[mask] / dv[mask])))
    return 1.0

def cfactorize(M):
    reg = 0.0
    I = np.eye(M.shape[0], dtype=M.dtype)
    for _ in range(6):
        try:
            return np.linalg.cholesky(M + reg * I)
        except np.linalg.LinAlgError:
            reg = 1e-12 if reg == 0.0 else reg * 10.0
    return np.linalg.cholesky(M + reg * I)

def csolve(L, rhs):
    y = np.linalg.solve(L, rhs)
    return np.linalg.solve(L.T, y)

def Solve(c, A, b, max_iter=200, use_meh=True):
    
    # define main matrices
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    c = np.asarray(c, dtype=float).reshape(-1)

    m, n = A.shape

    # basic checks
    tol_cert = 1e-8
    for i in range(m):
        if b[i] < -tol_cert and np.all(A[i, :] >= -tol_cert):
            return "infeasible", None, None

    if np.all(b >= -tol_cert):
        col_max = np.max(A, axis=0)
        if np.any((c > tol_cert) & (col_max <= tol_cert)):
            return "unbounded", None, None
    
    # scale problem
    ub = np.min(np.where(A > 0, b[:, None] / (A + 1e-12), np.inf), axis=0)
    s = np.where(np.isfinite(ub) & (ub > 1.0), ub, 1.0)
    A, c = A * s[None, :], c * s

    # initial state
    x, y = np.ones(n), np.ones(m),
    w, z = np.maximum(b - A @ x, 1.0), np.maximum(A.T @ y - c, 1.0)
    nb = np.linalg.norm(b, np.inf)
    nc = np.linalg.norm(c, np.inf)
    
    for _ in range(max_iter):
        
        # compute residuals
        rho = b - A @ x - w
        sigma = c - A.T @ y + z
        
        # current mu
        mu_bar = float(x @ z + y @ w) / (n + m)
        
        # if optimal
        if (
            np.linalg.norm(rho, np.inf) <= 1e-8 * (1.0 + nb) and
            np.linalg.norm(sigma, np.inf) <= 1e-8 * (1.0 + nc) and
            mu_bar <= 1e-8
        ):
            x_orig = x * s
            return "optimal", x_orig, float((c / s) @ x_orig)
        
        # diag matrices
        H, Q = z / x + 1e-12, w / y + 1e-12
        D = 1.0 / H

        # SPD eq
        M = (A * D) @ A.T
        M[np.arange(m), np.arange(m)] += Q
        L = cfactorize(M)
        
        # restore steps by residuals
        def solve_rhs(r_xz, r_yw):
            rhs = -rho + (r_yw / y) + A @ (D * (sigma + (r_xz / x)))
            dy = csolve(L, rhs)
            dx = D * (sigma + (r_xz / x) - A.T @ dy)
            dw = rho - A @ dx
            dz = A.T @ dy - sigma
            return dx, dw, dy, dz

        if use_meh:
            
            # mehorta correction
            r_xz_aff = -(x * z)
            r_yw_aff = -(y * w)
            dx_a, dw_a, dy_a, dz_a = solve_rhs(r_xz_aff, r_yw_aff)
            
            # step for aff
            th_aff = min(
                1.0,
                get_theta(x, dx_a),
                get_theta(w, dw_a),
                get_theta(y, dy_a),
                get_theta(z, dz_a),
            )
            x_aff = x + th_aff * dx_a
            w_aff = w + th_aff * dw_a
            y_aff = y + th_aff * dy_a
            z_aff = z + th_aff * dz_a
            
            # mu relative
            mu_aff = float((x_aff @ z_aff + y_aff @ w_aff) / (n + m))
            sig_c = max(min((mu_aff / mu_bar) ** 3, 1.0), 0.0)
        
            # second order rhs
            r_xz = sig_c * mu_bar - x * z - (dx_a * dz_a)
            r_yw = sig_c * mu_bar - y * w - (dy_a * dw_a)
            dx, dw, dy, dz = solve_rhs(r_xz, r_yw)
            
        else:
            # simple rhs solving
            mu = 0.1 * mu_bar
            r_xz = mu - x * z
            r_yw = mu - y * w
            dx, dw, dy, dz = solve_rhs(r_xz, r_yw)

        # select step
        th = 0.99 * min(
            1.0,
            get_theta(x, dx),
            get_theta(w, dw),
            get_theta(y, dy),
            get_theta(z, dz),
        )
        
        # ensure positive
        x = np.maximum(x + th * dx, 1e-30)
        w = np.maximum(w + th * dw, 1e-30)
        y = np.maximum(y + th * dy, 1e-30)
        z = np.maximum(z + th * dz, 1e-30)

    return "max_iter", None, None


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
    
    A_full = np.hstack([A, np.eye(m)])
    c_full = np.concatenate([c, np.zeros(m)])
    
    print("n =", n)
    print("m =", m)
    print("c =", c)
    print("A =", A)
    print("b =", b)

    print("Solving the linear program using the Primal Simplex method...\n")
    status, solution, objective = Solve(c_full, A_full, b)
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