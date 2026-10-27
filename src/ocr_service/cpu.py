"""Choose an inference CPU budget while keeping capacity for the host."""

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CPUPolicy:
    threads: int
    cpu_ids: list[int]
    available_threads: int
    quota: float | None


def cpu_quota(
    root: Path = Path("/sys/fs/cgroup"), membership: Path = Path("/proc/self/cgroup")
) -> float | None:
    """Read the tightest visible cgroup v1/v2 quota, including parent limits."""
    directories = {root, root / "cpu", root / "cpu,cpuacct"}
    try:
        for line in membership.read_text().splitlines():
            _, controllers, group = line.split(":", 2)
            if controllers and "cpu" not in controllers.split(","):
                continue
            # A cgroup namespace may hide the host path. Always inspect mount roots too.
            if ".." in Path(group).parts:
                continue
            bases = [root] if not controllers else [root / "cpu", root / "cpu,cpuacct"]
            for base in bases:
                directory = base / group.lstrip("/")
                while directory != base:
                    directories.add(directory)
                    directory = directory.parent
    except (OSError, ValueError):
        pass
    limits = []
    for directory in directories:
        for quota_file, period_file in (
            ("cpu.max", None),
            ("cpu.cfs_quota_us", "cpu.cfs_period_us"),
        ):
            try:
                if period_file:
                    quota, period = (
                        (directory / quota_file).read_text(),
                        (directory / period_file).read_text(),
                    )
                else:
                    quota, period = (directory / quota_file).read_text().split()
                if int(quota) > 0 and int(period) > 0:
                    limits.append(int(quota) / int(period))
            except (OSError, ValueError):
                continue
    return min(limits) if limits else None


def cpu_policy(
    requested: int | Literal["auto"],
    reserve_cores: int = 1,
    topology: Path = Path("/sys/devices/system/cpu"),
) -> CPUPolicy:
    try:
        available = sorted(os.sched_getaffinity(0))
    except (OSError, AttributeError):
        available = list(range(os.cpu_count() or 1))
    groups = {}
    for cpu in available:
        try:
            base = topology / f"cpu{cpu}" / "topology"
            key = (
                int((base / "physical_package_id").read_text()),
                int((base / "core_id").read_text()),
            )
            if min(key) < 0:
                raise ValueError("Unknown topology")
        except (OSError, ValueError):
            key = (-1, cpu)
        groups.setdefault(key, []).append(cpu)
    cores = list(groups.values())
    reserved = min(reserve_cores, max(0, len(cores) - 1))
    usable = cores[reserved:]
    # Spread work across physical cores before adding SMT siblings.
    selected = [
        core[index]
        for index in range(max(map(len, usable)))
        for core in usable
        if index < len(core)
    ]
    if len(cores) == 1 and reserve_cores:
        selected = selected[: max(1, len(selected) - 1)]
    quota = cpu_quota()
    budget = len(selected)
    if quota is not None:
        # Quotas measure logical CPU time; leave at least one CPU-equivalent when possible.
        budget = min(budget, max(1, math.floor(quota) - reserve_cores))
    threads = budget if requested == "auto" else min(requested, budget)
    return CPUPolicy(threads, selected[:threads], len(available), quota)


def apply_cpu_policy(policy: CPUPolicy, nice: int) -> None:
    """Apply only inside the isolated inference child, before importing native runtimes."""
    try:
        os.sched_setaffinity(0, policy.cpu_ids)
    except (OSError, AttributeError) as error:
        LOGGER.warning("Could not reserve CPUs via affinity: %s", error)
    try:
        # Never raise priority or accumulate niceness across repeated initialization.
        current = os.getpriority(os.PRIO_PROCESS, 0)
        if current < nice:
            os.setpriority(os.PRIO_PROCESS, 0, nice)
    except (OSError, AttributeError) as error:
        LOGGER.warning("Could not lower inference priority: %s", error)
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[name] = str(policy.threads)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_BLOCKTIME"] = "0"
    LOGGER.info(
        "Inference CPU budget: %s threads, CPUs %s, quota %s, nice %s",
        policy.threads,
        policy.cpu_ids,
        policy.quota,
        nice,
    )
