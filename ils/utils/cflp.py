from pathlib import Path


def get_path(data_path, problem, ptype="inst", suffix=".pkl", as_str=False):
    p = Path(data_path) / "cflp"
    p.mkdir(parents=True, exist_ok=True)

    _, n_facilities, n_customers = problem.split("_")

    p = p / f"{ptype}_f{n_facilities}_c{n_customers}{suffix}"
            
    if as_str:
        return str(p)
    return p
