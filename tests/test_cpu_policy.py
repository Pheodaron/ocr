import json
import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from ocr_service import cpu
from ocr_service.config import Settings
from ocr_service.schemas import PageResult


def topology(tmp_path, monkeypatch, cores):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(cores))
    for logical, (package, core) in cores.items():
        directory = tmp_path / f"cpu{logical}" / "topology"
        directory.mkdir(parents=True)
        (directory / "physical_package_id").write_text(str(package))
        (directory / "core_id").write_text(str(core))
    monkeypatch.setattr(cpu, "cpu_quota", lambda: None)


def test_auto_uses_smt_but_reserves_all_siblings_of_a_physical_core(tmp_path, monkeypatch):
    topology(tmp_path, monkeypatch, {i: (0, i % 4) for i in range(8)})
    policy = cpu.cpu_policy("auto", 1, tmp_path)
    assert policy.cpu_ids == [1, 2, 3, 5, 6, 7]
    assert policy.threads == 6
    assert policy.available_threads == 8
    assert cpu.cpu_policy("auto", 2, tmp_path).cpu_ids == [2, 3, 6, 7]
    # Explicit settings are upper bounds; physical cores are used before SMT siblings.
    assert cpu.cpu_policy(2, 1, tmp_path).cpu_ids == [1, 2]
    assert cpu.cpu_policy(128, 1, tmp_path).threads == 6


def test_cpuset_and_package_identity_are_respected(tmp_path, monkeypatch):
    topology(tmp_path, monkeypatch, {1: (0, 0), 3: (0, 0), 8: (1, 0), 9: (1, 0)})
    assert cpu.cpu_policy("auto", 1, tmp_path).cpu_ids == [8, 9]


@pytest.mark.parametrize("threads,expected", [(1, 1), (2, 1)])
def test_small_hosts_keep_at_least_one_inference_thread(tmp_path, monkeypatch, threads, expected):
    topology(tmp_path, monkeypatch, {i: (0, 0) for i in range(threads)})
    assert cpu.cpu_policy("auto", 2, tmp_path).threads == expected


def test_cpu_quota_caps_auto_and_manual_thread_budgets(tmp_path, monkeypatch):
    topology(tmp_path, monkeypatch, {i: (0, i) for i in range(16)})
    monkeypatch.setattr(cpu, "cpu_quota", lambda: 4.5)
    assert cpu.cpu_policy("auto", 1, tmp_path).threads == 3
    assert cpu.cpu_policy(16, 1, tmp_path).threads == 3
    monkeypatch.setattr(cpu, "cpu_quota", lambda: 0.5)
    assert cpu.cpu_policy("auto", 1, tmp_path).threads == 1


def test_cgroup_v2_uses_tightest_parent_quota(tmp_path):
    root = tmp_path / "cgroup"
    child = root / "parent" / "child"
    child.mkdir(parents=True)
    (root / "cpu.max").write_text("800000 100000")
    (root / "parent" / "cpu.max").write_text("150000 100000")
    (child / "cpu.max").write_text("max 100000")
    membership = tmp_path / "membership"
    membership.write_text("0::/parent/child\n")
    assert cpu.cpu_quota(root, membership) == 1.5
    membership.write_text("0::/unavailable/namespace/path\n")
    assert cpu.cpu_quota(root, membership) == 8


def test_cgroup_v1_and_unlimited_or_missing_quota(tmp_path):
    root = tmp_path / "cgroup"
    child = root / "cpu,cpuacct" / "worker"
    child.mkdir(parents=True)
    membership = tmp_path / "membership"
    membership.write_text("2:cpu,cpuacct:/worker\n")
    (child / "cpu.cfs_quota_us").write_text("250000")
    (child / "cpu.cfs_period_us").write_text("100000")
    assert cpu.cpu_quota(root, membership) == 2.5
    (child / "cpu.cfs_quota_us").write_text("-1")
    assert cpu.cpu_quota(root, membership) is None
    assert cpu.cpu_quota(tmp_path / "missing", membership) is None


def test_missing_topology_falls_back_to_allowed_logical_cpus(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {2, 4, 8})
    monkeypatch.setattr(cpu, "cpu_quota", lambda: None)
    assert cpu.cpu_policy("auto", 1, tmp_path).cpu_ids == [4, 8]


def test_auto_settings_round_trip_and_manual_lighton_limit_can_exceed_eight():
    settings = Settings(_env_file=None)
    snapshot = settings.recognition_config()
    assert snapshot["cpu_threads"] == snapshot["lighton_small_threads"] == "auto"
    assert snapshot["cpu_reserve_cores"] == 1
    assert snapshot["cpu_nice"] == 10
    restored = settings.restore_recognition_config(snapshot)
    assert restored.cpu_threads == "auto"
    assert Settings(_env_file=None, lighton_small_threads=24).lighton_small_threads == 24
    assert Settings(_env_file=None, cpu_threads="12").cpu_threads == 12
    for value in (0, -1, 1025, "invalid"):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, cpu_threads=value)


def test_affinity_priority_and_thread_environment_change_only_in_child():
    parent_ids = sorted(os.sched_getaffinity(0))
    parent_nice = os.getpriority(os.PRIO_PROCESS, 0)
    selected = parent_ids[-1:]
    script = f"""
import json, os
from ocr_service.cpu import CPUPolicy, apply_cpu_policy
apply_cpu_policy(CPUPolicy(1, {selected!r}, {len(parent_ids)}, None), 10)
apply_cpu_policy(CPUPolicy(1, {selected!r}, {len(parent_ids)}, None), 10)
print(json.dumps({{
    "cpus": sorted(os.sched_getaffinity(0)),
    "nice": os.getpriority(os.PRIO_PROCESS, 0),
    "threads": os.environ["OMP_NUM_THREADS"],
    "limit": os.environ["OMP_THREAD_LIMIT"],
    "wait": os.environ["OMP_WAIT_POLICY"],
}}))
"""
    child = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    actual = json.loads(child.stdout)
    assert actual == {
        "cpus": selected,
        "nice": max(10, parent_nice),
        "threads": "1",
        "limit": "1",
        "wait": "PASSIVE",
    }
    assert sorted(os.sched_getaffinity(0)) == parent_ids
    assert os.getpriority(os.PRIO_PROCESS, 0) == parent_nice


def test_nemotron_is_unavailable_for_new_jobs(client, pdf_bytes):
    migrated = Settings(_env_file=None, enabled_backends="paddle,nemotron,jina,glm_ocr")
    assert migrated.backends == ["paddle", "glm_ocr"]
    assert "nemotron" not in {item["id"] for item in client.get("/api/config").json()["backends"]}
    response = client.post(
        "/api/documents", files={"file": ("document.pdf", pdf_bytes)}, data={"backend": "nemotron"}
    )
    assert response.status_code == 422
    assert "jina" not in {item["id"] for item in client.get("/api/config").json()["backends"]}
    assert (
        client.post(
            "/api/documents", files={"file": ("document.pdf", pdf_bytes)}, data={"backend": "jina"}
        ).status_code
        == 422
    )
    # Removing the runtime must not remove or hide previously saved recognition results.
    db = client.app.state.db
    old_id = "f" * 32
    db.create_job(old_id, "archive.pdf", "0" * 64, 1, {}, backend="nemotron")
    db.claim_job(["nemotron"])
    db.save_page(old_id, PageResult(number=1, text="Сохранённый результат"))
    db.finish_job(old_id)
    archived = client.get(f"/api/documents/{old_id}/result")
    assert archived.status_code == 200
    assert archived.json()["pages"][0]["text"] == "Сохранённый результат"
