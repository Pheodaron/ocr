"""CPU placement for the LightOn 0.8B experiment."""

import logging
import os
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def physical_cpu_ids(
    limit: int, topology: Path = Path("/sys/devices/system/cpu")
) -> list[int] | None:
    """Select one allowed logical CPU per physical core, across CPU packages."""
    try:
        available = sorted(os.sched_getaffinity(0))
        cores = {}
        for cpu in available:
            base = topology / f"cpu{cpu}" / "topology"
            package = int((base / "physical_package_id").read_text())
            core = int((base / "core_id").read_text())
            if package < 0 or core < 0:
                raise ValueError("CPU topology unavailable")
            cores.setdefault((package, core), cpu)
        selected = list(cores.values())[:limit]
        if not selected:
            raise ValueError("No allowed physical cores")
        return selected
    except (OSError, ValueError, AttributeError) as error:
        LOGGER.warning("Physical CPU affinity unavailable; keeping thread limit only: %s", error)
        return None
