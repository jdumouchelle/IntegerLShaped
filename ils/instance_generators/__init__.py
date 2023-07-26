from .cflp import CFLPGenerator
from .sslp import SSLPGenerator

def factory_instance_generator(problem, params):
    if "cflp" in problem:
        print("Loading CFLP instance generator...")
        return CFLPGenerator(**params)
    elif "sslp" in problem:
        print("Loading SSLP instance generator...")
        return SSLPGenerator(**params)
    else:
        raise ValueError("Invalid problem type!")