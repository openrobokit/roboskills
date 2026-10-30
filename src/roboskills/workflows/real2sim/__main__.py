import tyro

from roboskills.workflows.real2sim import run


if __name__ == "__main__":  # not in the retry sets' spawned processes
    tyro.cli(run)
