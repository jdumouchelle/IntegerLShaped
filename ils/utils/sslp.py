from pathlib import Path


def get_path(data_path, problem, ptype="inst", suffix=".pkl", as_str=False):
    p = Path(data_path) / "sslp"
    p.mkdir(parents=True, exist_ok=True)

    _, n_locations, n_clients = problem.split("_")

    p = p / f"{ptype}_l{n_locations}_c{n_clients}{suffix}"

    if as_str:
        return str(p)
    return p
