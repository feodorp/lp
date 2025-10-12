import numpy as np

eps = 1e-9

def lu_factor(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(float)
    
    for k in range(n-1):
        if abs(U[k, k]) < eps:
            continue
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
    
    return L, U


def forward(L, b):
    m = L.shape[0]
    x = b.copy()
    for i in range(m):
        for j in range(i):
            x[i] -= L[i, j] * x[j]
    return x

def back(U, b):
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        denom = U[i, i]
        if abs(denom) < eps:
            x[i] = 0.0
        else:
            x[i] = (b[i] - U[i, i+1:].dot(x[i+1:])) / denom
    return x

def back_t(LT, b):
    m = LT.shape[0]
    x = b.copy()
    for i in range(m-1, -1, -1):
        for j in range(i+1, m):
            x[i] -= LT[j, i] * x[j]
    return x

def forward_t(UT, b):
    m = UT.shape[0]
    x = b.copy()
    for i in range(m):
        if abs(UT[i, i]) < eps:
            x[i] = 0.0
        for j in range(i):
            x[i] -= UT[j, i] * x[j]
        x[i] /= UT[i, i]
    return x

def solve_B_with_LU(L, U, b):
    y = forward(L, b)
    x = back(U, y)
    return x

def solve_Bt_with_LU(L, U, b):
    x = forward_t(U, b)
    y = back_t(L, x)
    return y


def update_lu(L, U, p, aj):
    m = L.shape[0]
    
    # 2
    v = forward(L, aj)
    
    # 3
    H = U.copy()
    H[:, p] = v
    
    # 4
    L_update = np.eye(m)
    
    for i in range(m-2, p-1, -1):
        if abs(H[i, p]) < eps:
            continue
            
        pivot_row = i
        if abs(H[i+1, p]) > abs(H[i, p]):
            pivot_row = i + 1
        
        if pivot_row != i:
            H[[i, pivot_row], :] = H[[pivot_row, i], :]
            L_update[[i, pivot_row], :] = L_update[[pivot_row, i], :]
        
        if abs(H[i, p]) < eps:
            continue
            
        factor = H[i+1, p] / H[i, p]
        H[i+1, p:] = H[i+1, p:] - factor * H[i, p:]
        L_update[i+1, i] = factor
    
    U_new = H
    L_new = L @ L_update
    
    return L_new, U_new

