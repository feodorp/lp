import scipy.linalg as la
import numpy as np

class FactorizedMatrix:
    def __init__(self, A):
        self.A = A

        self.LU    = None
        self.pivot = None

    def init_factorization(self):
        self.LU, self.pivot = la.lu_factor(self.A)

    def solve(self, b):
        """Solves Ax = b"""
        # Ax = (PLU)x = b => (LU)x = P.T @ b = b_hat
        # Ux = y => Ly = b_hat  => Ax = b
        assert b.shape[1] == 1
        x = la.lu_solve((self.LU, self.pivot), b)
        return x
