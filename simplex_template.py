import numpy as np
import sys
import argparse

# global vars
eps = 0.00001
upper_bound=1e8


def DualSimplexMethod(c, A, b):
    c = np.asarray(c, dtype=float).flatten()
    A = np.array(A, dtype=float)
    b = np.asarray(b, dtype=float).flatten()

    num_A, num_original_vars = A.shape
    total_vars = num_original_vars + num_A

    identity_m = np.eye(num_A, dtype=float)
    ext_A = np.hstack([A, identity_m])
    ext_c = np.concatenate([c, np.zeros(num_A, dtype=float)])

    basic_vars = list(range(num_original_vars, num_original_vars + num_A))
    nbasic_vars = list(range(num_original_vars))

    bounds = np.full(total_vars, upper_bound, dtype=float)

    directions = np.zeros(total_vars, dtype=float)

    N_m = ext_A[:, nbasic_vars]
    c_nbasic = ext_c[nbasic_vars]

    dual_vars = np.zeros(num_A)
    reduced_costs_nbasic = c_nbasic - N_m.T @ dual_vars

    x_nbasic = np.zeros(len(nbasic_vars))
    reduced_costs_directional = np.zeros(len(nbasic_vars))

    for idx, var_index in enumerate(nbasic_vars):
        if reduced_costs_nbasic[idx] <= eps:
            directions[var_index] = 1.0
            x_nbasic[idx] = 0.0
        else:
            directions[var_index] = -1.0
            x_nbasic[idx] = bounds[var_index]
        reduced_costs_directional[idx] = directions[var_index] * reduced_costs_nbasic[idx]


    for iteration in range(1000):
        basis_m = ext_A[:, basic_vars]
        nbasic_m = ext_A[:, nbasic_vars]
        c_basic = ext_c[basic_vars]

        try:
            dual_vars = np.linalg.solve(basis_m.T, c_basic)
        except np.linalg.LinAlgError:
            return "infeasible", None, None

        reduced_costs_nbasic = ext_c[nbasic_vars] - nbasic_m.T @ dual_vars
        reduced_costs_directional = np.empty(len(nbasic_vars))
        x_nbasic = np.empty(len(nbasic_vars))

        for idx, var_index in enumerate(nbasic_vars):
            reduced_costs_directional[idx] = directions[var_index] * reduced_costs_nbasic[idx]
            x_nbasic[idx] = 0.0 if directions[var_index] > 0 else bounds[var_index]

        try:
            x_basic = np.linalg.solve(basis_m, b - nbasic_m @ x_nbasic)
        except np.linalg.LinAlgError:
            return "infeasible", None, None

        is_feasible = np.all(x_basic >= -eps)
        is_dual_feasible = np.all(reduced_costs_directional <= eps)

        if is_feasible and is_dual_feasible:
            solution_vector = np.zeros(total_vars)
            for row_idx, var_idx in enumerate(basic_vars):
                solution_vector[var_idx] = x_basic[row_idx]
            for col_idx, var_idx in enumerate(nbasic_vars):
                solution_vector[var_idx] = x_nbasic[col_idx]

            original_solution = solution_vector[:num_original_vars]
            objective_value = float(c @ original_solution)

            if np.any(original_solution >= upper_bound - 1e-6 * upper_bound):
                return "unbounded", None, None

            return "optimal", original_solution, objective_value

        infeasible_rows = [i for i in range(num_A) if x_basic[i] < -eps]
        leaving_row = min(infeasible_rows, key=lambda i: basic_vars[i])
        leaving_var = basic_vars[leaving_row]

        try:
            basis_inverse = np.linalg.inv(basis_m)
        except np.linalg.LinAlgError:
            return "infeasible", None, None

        pivot_row = basis_inverse[leaving_row, :]
        tableau_row = pivot_row @ ext_A

        sigma_values = np.empty(len(nbasic_vars))
        for idx, var_idx in enumerate(nbasic_vars):
            sigma_values[idx] = -tableau_row[var_idx] * directions[var_idx]

        candidate_indices = []
        ratio_values = []

        for idx in range(len(nbasic_vars)):
            if sigma_values[idx] > eps:
                ratio = -reduced_costs_directional[idx] / sigma_values[idx]
                candidate_indices.append(idx)
                ratio_values.append(ratio)

        if not candidate_indices:
            return "infeasible", None, None

        min_ratio = min(ratio_values)
        best_candidates = [
            idx for idx, ratio in zip(candidate_indices, ratio_values)
            if ratio <= min_ratio + eps
        ]

        entering_position = min(best_candidates, key=lambda i: nbasic_vars[i])
        entering_var = nbasic_vars[entering_position]

        basic_vars[leaving_row] = entering_var
        nbasic_vars[entering_position] = leaving_var

        directions[leaving_var] = 1.0

    return "iterations_limit", None, None


def Solve(c, A, b):
    return DualSimplexMethod(c, A, b)

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
    print("n =", n)
    print("m =", m)
    print("c =", c)
    print("A =", A)
    print("b =", b)

    print("Solving the linear program using the Dual Simplex method...\n")
    status, solution, objective = Solve(c, A, b)
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