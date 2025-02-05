import numpy as np
from scipy.optimize import minimize

def nested_minimizer_example():
    # Define an inner function that depends on both x and y
    def f_xy(x, y):
        return (x - 2)**2 + (y + 3*x)**2

    # Inner minimization: for a given x, find the y that minimizes f_xy
    def inner_minimize_y(x):
        def objective(y):
            return f_xy(x, y)
        # Minimize w.r.t. y
        result = minimize(objective, 0.0)  # start at y=0
        return result.fun, result.x[0]

    # Outer objective: for each x, the objective is the minimal value over y
    def outer_objective(x):
        best_val, _ = inner_minimize_y(x)
        return best_val

    # Now minimize outer_objective w.r.t. x
    result_outer = minimize(outer_objective, 0.0)  # start at x=0
    x_opt = result_outer.x[0]
    f_opt, y_opt = inner_minimize_y(x_opt)

    print(f"Optimal x: {x_opt:.3f}")
    print(f"Optimal y: {y_opt:.3f}")
    print(f"Optimal function value: {f_opt:.3f}")

if __name__ == "__main__":
    nested_minimizer_example()
