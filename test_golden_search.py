def golden_section_search(f, a, b, tol=1e-5):
    """
    Perform Golden-section search to find the minimum of a unimodal function f in [a, b].

    Parameters:
    f   : function - The function to minimize.
    a   : float - The lower bound of the interval.
    b   : float - The upper bound of the interval.
    tol : float - The tolerance for convergence.

    Returns:
    float - The estimated minimum point.
    """
    phi = (1 + 5 ** 0.5) / 2  # Golden ratio
    resphi = 2 - phi  # 1 - (phi - 1)

    # Initial points
    c = a + resphi * (b - a)
    d = b - resphi * (b - a)
    
    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
        else:
            a = c
        
        # Recompute new points
        c = a + resphi * (b - a)
        d = b - resphi * (b - a)

    return (a + b) / 2  # Approximate minimum point

# Example usage
if __name__ == "__main__":
    f = lambda x: (x - 2) ** 2  # Example function with a minimum at x=2
    minimum = golden_section_search(f, 0, 5)
    print(f"Estimated minimum at x = {minimum:.5f}")
