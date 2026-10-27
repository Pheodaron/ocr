"""Wall timings and sampled resource use of an isolated inference process tree."""

import os
import platform
import threading
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def timed_stage(stages: list[dict], name: str, **details):
    stage = {"stage": name, "details": details, "status": "completed"}
    started = time.monotonic()
    try:
        yield details
    except Exception as error:
        stage.update(status="failed", error=str(error)[:1000])
        raise
    finally:
        stage["seconds"] = round(time.monotonic() - started, 6)
        stages.append(stage)


def process_tree_snapshot() -> dict:
    """Read CPU counters/RSS for this process and live descendants on Linux."""
    result = {}
    pending = [os.getpid()]
    visited = set()
    ticks = os.sysconf("SC_CLK_TCK")
    page_size = os.sysconf("SC_PAGE_SIZE")
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        base = Path(f"/proc/{pid}")
        try:
            fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
            result[(pid, fields[19])] = (
                (int(fields[11]) + int(fields[12])) / ticks,
                int(fields[21]) * page_size,
            )
            for children in (base / "task").glob("*/children"):
                try:
                    pending.extend(map(int, children.read_text().split()))
                except (OSError, ValueError):
                    continue
        except (OSError, ValueError, IndexError):
            continue
    return result


class ResourceMonitor:
    """CPU time and summed RSS, sampled every 100 ms (not an exclusive-memory peak)."""

    def __enter__(self):
        self.started = time.monotonic()
        self.last = process_tree_snapshot()
        self.cpu_seconds = 0.0
        self.rss_peak = sum(value[1] for value in self.last.values())
        self.samples = int(bool(self.last))
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        snapshot = process_tree_snapshot()
        if snapshot:
            self.samples += 1
            self.cpu_seconds += sum(
                max(0, value[0] - self.last.get(key, (0, 0))[0]) for key, value in snapshot.items()
            )
            self.last.update(snapshot)
            self.rss_peak = max(self.rss_peak, sum(value[1] for value in snapshot.values()))

    def _run(self):
        while not self.stop.wait(0.1):
            self._sample()

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join()
        self._sample()
        self.wall_seconds = time.monotonic() - self.started

    def metrics(self) -> dict:
        return {
            "cpu_seconds": round(self.cpu_seconds, 6) if self.samples else None,
            "sampled_tree_rss_peak_bytes": self.rss_peak if self.samples else None,
            "samples": self.samples,
            "sample_interval_seconds": 0.1,
            "scope": "inference process and observed live descendants",
        }


def host_resources() -> dict:
    try:
        cpu_model = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        cpu_model = platform.processor() or None
    return {
        "architecture": platform.machine(),
        "cpu_model": cpu_model,
        "device": "cpu",
    }
