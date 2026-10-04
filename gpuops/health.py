"""The Slurm HealthCheckProgram.

slurmd runs it as root on every node every HealthCheckInterval seconds (with SLURMD_NODENAME set).
It reads the node's GPU health, applies the policy and acts through scontrol:

    cordon      -> drain; running jobs finish
    drain       -> drain, and requeue the jobs running on the node (a job on a GPU that fell off
                   the bus does not crash cleanly: it hangs in a collective until something kills it)
    quarantine  -> same as drain, with a reason that `gpuops validate` refuses to clear

It never resumes a node: putting a node back is `gpuops validate`, after a repair. A node drained by
someone else (reason not starting with "gpu-health:") keeps its reason; its jobs are still requeued
when the GPU fault breaks them. Each decision that is not "none" is appended to a JSON Lines log.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

from . import policy
from .slurm import NodeInfo, Slurm, SlurmError
from .sources import now

PREFIX = "gpu-health:"


def reason_action(reason: str) -> str | None:
    """'gpu-health:drain XID 79 on GPU 1 (...)' -> 'drain'; None if the reason is not ours."""
    if not reason.startswith(PREFIX):
        return None
    a = reason[len(PREFIX):].split(" ", 1)[0]
    return a if a in policy.RANK else None


def decide(node: NodeInfo, findings: list[policy.Finding]) -> dict:
    """What to do, without doing it: {"drain": reason or None, "requeue": bool, "note": str}."""
    action = policy.worst(findings)
    plan = {"action": action, "drain": None, "requeue": False, "note": ""}
    if policy.RANK[action] < policy.RANK["cordon"]:
        return plan
    plan["requeue"] = policy.RANK[action] >= policy.RANK["drain"]
    ours = reason_action(node.reason)
    reason = f"{PREFIX}{action} {findings[0].text()}"
    if node.drained and ours is None and node.reason:
        plan["note"] = f"already drained by someone else ({node.reason}); reason left as is"
    elif node.drained and ours is not None and policy.RANK[ours] >= policy.RANK[action]:
        plan["note"] = "already drained for this"
    else:
        plan["drain"] = reason
    return plan


def run(node_name: str, source, slurm: Slurm, log_path: Path | None, temp_cordon_c: float) -> dict:
    gpus = source.read(node_name)
    findings = policy.evaluate(gpus, temp_cordon_c)
    node = slurm.node(node_name)
    plan = decide(node, findings)
    requeued, errors = [], []
    if plan["drain"]:
        slurm.drain(node_name, plan["drain"])
    if plan["requeue"]:
        for job in slurm.running_jobs(node_name):
            try:
                slurm.requeue(job)
                requeued.append(job)
            except SlurmError as e:  # e.g. a job submitted with --no-requeue: Slurm keeps it, a human decides
                errors.append(str(e))
    record = {"time": now(), "node": node_name, "action": plan["action"], "state": node.state,
              "findings": [{"gpu": f.gpu, "action": f.action, "what": f.what, "detail": f.detail, "time": f.time}
                           for f in findings],
              "drained": plan["drain"], "requeued": requeued, "note": plan["note"]}
    if errors:
        record["errors"] = errors
    changed = bool(plan["drain"] or requeued or errors)
    if log_path and (changed or (plan["action"] != "none" and record["findings"] != _last_findings(log_path))):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
    return record


def _last_findings(log_path: Path):
    """Findings of the last logged record: a fault that is already logged and acted on is not logged
    again every interval."""
    try:
        lines = log_path.read_text().strip().splitlines()
        return json.loads(lines[-1])["findings"] if lines else None
    except (OSError, ValueError, KeyError):
        return None


def node_name() -> str:
    return os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
