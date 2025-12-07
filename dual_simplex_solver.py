import numpy as np

eps = 0.00001
U = 1e12


class DualSimplexSolver:
    def __init__(self, c, A, b, n_orig):
        self.c_orig = c.copy()
        self.n_orig = n_orig
        self.m = A.shape[0]
        
        self.A = np.hstack([A, np.eye(self.m)])
        self.c = np.concatenate([c, np.zeros(self.m)])
        self.b = b.copy()
        
        self.basis_indices = np.arange(n_orig, n_orig + self.m)
        self.nbasis_indices = np.arange(0, n_orig)
        
        self.B_inv = None
        self.iteration_count = 0
        self.max_iterations = 10000
    
    def _extract_submatrices(self):
        B = self.A[:, self.basis_indices]
        N = self.A[:, self.nbasis_indices]
        return B, N
    
    def _compute_reduced_costs(self, B, N):
        c_B = self.c[self.basis_indices]
        c_N = self.c[self.nbasis_indices]
        z_N = (self.B_inv @ N).T @ c_B - c_N
        return -z_N
    
    def _initialize_nonbasic_vars(self, r_n):
        x_n = np.zeros(self.n_orig)
        d = np.ones(self.n_orig)
        
        is_upper = r_n > eps
        x_n[is_upper] = U
        d[is_upper] = -1.0
        
        return x_n, d
    
    def _compute_basic_vars(self, N, x_n):
        return self.B_inv @ (self.b - N @ x_n)
    
    def _find_entering_variable(self, x_b):
        i = np.argmin(x_b)
        return i if x_b[i] < -eps else -1
    
    def _compute_step_direction(self, N, i, d):
        a = self.B_inv @ N
        return -a[i, :].ravel() * d
    
    def _find_leaving_variable(self, sigma, r):
        candidates = np.where(sigma > eps)[0]
        
        if len(candidates) == 0:
            return -1
        
        theta_vals = -r[candidates] / sigma[candidates]
        j = candidates[np.argmin(theta_vals)]
        
        return j
    
    def _perform_pivot(self, i, j):
        self.basis_indices[i], self.nbasis_indices[j] = self.nbasis_indices[j], self.basis_indices[i]
        
        B, N = self._extract_submatrices()
        a_i = B[:, i].reshape(-1, 1)
        a_j = N[:, j].reshape(-1, 1)
        u = a_i - a_j
        v = np.zeros((self.m, 1))
        v[i, 0] = 1
        
        denom = 1 + v.T @ self.B_inv @ u
        self.B_inv = self.B_inv - (self.B_inv @ u) @ (v.T @ self.B_inv) / denom
    
    def _construct_solution(self, x_b, x_n):
        x_full = np.zeros(self.n_orig + self.m)
        x_full[self.basis_indices] = x_b
        x_full[self.nbasis_indices] = x_n
        return x_full[:self.n_orig]
    
    def solve(self):
        B, _ = self._extract_submatrices()
        self.B_inv = np.linalg.inv(B)
        
        while self.iteration_count < self.max_iterations:
            self.iteration_count += 1
            
            B, N = self._extract_submatrices()
            r_n = self._compute_reduced_costs(B, N)
            x_n, d = self._initialize_nonbasic_vars(r_n)
            x_b = self._compute_basic_vars(N, x_n)
            r = d * r_n
            
            i = self._find_entering_variable(x_b)
            if i == -1:
                x_solution = self._construct_solution(x_b, x_n)
                
                if np.any(x_solution == U):
                    return "unbounded", None, None
                
                obj = self.c_orig @ x_solution
                return "optimal", x_solution, obj
            
            sigma = self._compute_step_direction(N, i, d)
            
            if not np.any(sigma > eps):
                return "infeasible", None, None
            
            j = self._find_leaving_variable(sigma, r)
            
            self._perform_pivot(i, j)
        
        return "max_iterations", None, None
