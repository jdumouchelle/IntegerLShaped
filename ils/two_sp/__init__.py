from .two_sp import TwoStageStocProg


def factory_two_sp(problem, inst, sampler=None):
    if 'cflp' in problem:
        from .cflp import CFLP
        return CFLP(inst)

    elif 'sslp' in problem:
        from .sslp import SSLP
        return SSLP(inst)

    else:
        raise Exception(f"ils.two_sp not defined for problem class {problem}")
