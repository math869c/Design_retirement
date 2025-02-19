from numba import njit

USE_JIT = True  # Set to False to disable JIT for debugging

def jit_if_enabled(parallel=False , fastmath=False):
    """ Apply @njit only if USE_JIT is True """
    return njit(parallel=parallel) if USE_JIT else (lambda f: f)
