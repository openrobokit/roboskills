"""Container execution preserves explicit host resource restrictions."""

from roboskills.eval import runtime


def test_container_preserves_noncontiguous_parent_cpu_affinity(monkeypatch):
    monkeypatch.setattr(runtime, "docker_has", lambda image: True)
    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda pid: {7, 2}, raising=False)
    command = runtime.contained("image", ["python", "worker.py"])
    assert command[command.index("--cpuset-cpus") + 1] == "2,7"
    assert command[-3:] == ["python", "image", "worker.py"]


def test_container_remains_available_without_a_host_affinity_api(monkeypatch):
    monkeypatch.setattr(runtime, "docker_has", lambda image: True)
    monkeypatch.delattr(runtime.os, "sched_getaffinity", raising=False)
    assert "--cpuset-cpus" not in runtime.contained("image", ["python"])
