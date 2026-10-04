"""The few Slurm commands this project needs, behind one class so the tests can replace them."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


class SlurmError(RuntimeError):
    pass


@dataclass
class NodeInfo:
    name: str
    state: str  # sinfo long state: idle, mixed, allocated, draining, drained, down, maint, reserved...
    reason: str

    @property
    def drained(self) -> bool:
        return self.state.rstrip("*~#!%$@^-").startswith("drain")


class Slurm:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.calls: list[list[str]] = []  # changes made, for the health log

    def _run(self, *args: str, change: bool = False, check: bool = True) -> str:
        if change:
            self.calls.append(list(args))
            if self.dry_run:
                return ""
        p = subprocess.run(list(args), capture_output=True, text=True, timeout=30)
        if check and p.returncode != 0:
            raise SlurmError(f"{' '.join(args)}: {p.stderr.strip() or p.stdout.strip()}")
        return p.stdout

    # --- reads
    def node(self, name: str) -> NodeInfo:
        out = self._run("sinfo", "-h", "-N", "-n", name, "-o", "%N|%T|%E").strip().splitlines()
        if not out:
            raise SlurmError(f"unknown node {name}")
        n, state, reason = out[0].split("|", 2)
        return NodeInfo(n, state, "" if reason == "none" else reason)

    def nodes(self) -> list[NodeInfo]:
        rows = self._run("sinfo", "-h", "-N", "-o", "%N|%T|%E").strip().splitlines()
        seen, out = set(), []
        for r in rows:  # a node in two partitions is listed twice
            n, state, reason = r.split("|", 2)
            if n not in seen:
                seen.add(n)
                out.append(NodeInfo(n, state, "" if reason == "none" else reason))
        return out

    def running_jobs(self, node: str) -> list[str]:
        return self._run("squeue", "-h", "-w", node, "-t", "RUNNING", "-o", "%A").split()

    # --- changes
    def drain(self, node: str, reason: str) -> None:
        self._run("scontrol", "update", f"NodeName={node}", "State=DRAIN", f"Reason={reason}", change=True)

    def resume(self, node: str) -> None:
        self._run("scontrol", "update", f"NodeName={node}", "State=RESUME", change=True)

    def requeue(self, job: str) -> None:
        self._run("scontrol", "requeue", job, change=True)

    def create_reservation(self, name: str, node: str, minutes: int) -> None:
        self._run("scontrol", "create", "reservation", f"ReservationName={name}", f"Nodes={node}",
                  "StartTime=now", f"Duration={minutes}", "Users=root", "Flags=MAINT,IGNORE_JOBS", change=True)

    def delete_reservation(self, name: str) -> None:
        self._run("scontrol", "delete", f"ReservationName={name}", change=True, check=False)

    def submit(self, *args: str) -> str:
        return self._run("sbatch", "--parsable", *args, change=True).strip().split(";")[0]

    def job(self, job: str) -> tuple[str, int | None]:
        """(state, exit code or None while it has not ended), from `scontrol show job`."""
        out = self._run("scontrol", "show", "job", "-o", job)
        f = dict(kv.split("=", 1) for kv in out.split() if "=" in kv)
        state = f.get("JobState", "UNKNOWN")
        ended = state not in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "SUSPENDED")
        return state, int(f.get("ExitCode", "0:0").split(":")[0]) if ended else None

    def cancel(self, job: str) -> None:
        self._run("scancel", job, change=True, check=False)
