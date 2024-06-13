from .ils import IntegerLShaped

def factory_ils(problem, two_sp, scenarios):
    if 'cflp' in problem:
        from .cflp import CFLPIntegerLShaped
        return CFLPIntegerLShaped(two_sp, scenarios)

    elif 'sslp' in problem:
        from .sslp import SSLPIntegerLShaped
        return SSLPIntegerLShaped(two_sp, scenarios)

    else:
        raise Exception(f"ils.ils not defined for problem class {problem}")