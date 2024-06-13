import pickle
import numpy as np

def factory_get_path(problem):
    if 'cflp' in problem:
        from .cflp import get_path
        return get_path

    elif 'sslp' in problem:
        from .sslp import get_path
        return get_path

    else:
        raise Exception(f"ils.utils not defined for problem class {problem}")
