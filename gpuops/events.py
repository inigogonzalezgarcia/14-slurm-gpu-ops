"""Slurm accounting + the job's own log + the health-check log -> the event log of repo 12.

    sacct -D (every requeued run of the job)    start, nodes_ready (start of each run), end
    <runs>/<job>/events.jsonl (the job)          checkpoint_start / checkpoint_end, running
    <health>/*.jsonl (gpu-health)                interrupt (time and cause of the GPU fault),
                                                 detected (when the health check requeued the job)

The output is read by `python -m goodput analyze` from repo 12 (gpu-goodput-mtbi).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SACCT_FIELDS = "JobIDRaw,JobName,State,Start,End,NNodes,NodeList,AllocTRES,ExitCode"
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"  # passed to sacct as SLURM_TIME_FORMAT, with TZ=UTC
SLACK = timedelta(seconds=5)  # the health-log line is written just after the requeue it records


@dataclass
class Instance:
    job: str
    name: str
    state: str
    start: datetime
    end: datetime | None
    nnodes: int
    nodes: str
    gpus: int


def parse_time(s: str) -> datetime | None:
    if s in ("", "Unknown", "None"):
        return None
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_sacct(text: str) -> list[Instance]:
    """`sacct -D -X -P -n -o <SACCT_FIELDS>` output, one line per run of the job, oldest first."""
    out = []
    for line in text.strip().splitlines():
        f = line.split("|")
        if len(f) != len(SACCT_FIELDS.split(",")):
            raise ValueError(f"unexpected sacct line: {line!r}")
        job, name, state, start, end, nnodes, nodes, tres, _ = f
        start_t = parse_time(start)
        if start_t is None:
            continue  # a pending record that never ran
        m = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", tres)
        out.append(Instance(job, name, state.split()[0], start_t, parse_time(end), int(nnodes), nodes,
                            int(m.group(1)) if m else 0))
    out.sort(key=lambda i: i.start)
    return out


def run_sacct(job: str) -> str:
    env = dict(os.environ, SLURM_TIME_FORMAT=TIME_FORMAT, TZ="UTC")
    return subprocess.run(["sacct", "-j", job, "-D", "-X", "-P", "-n", "-o", SACCT_FIELDS],
                          capture_output=True, text=True, check=True, env=env, timeout=30).stdout


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def health_records(health_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(health_dir.glob("*.jsonl")):
        recs.extend(read_jsonl(p))
    return recs


def cause_of(finding: dict) -> str:
    what = finding["what"]
    if what.startswith("XID "):
        return "gpu_xid" + what.split()[1]
    return "gpu_" + re.sub(r"\W+", "_", what.lower()).strip("_")


def build(instances: list[Instance], job_events: list[dict], health: list[dict], name: str | None = None) -> list[dict]:
    if not instances:
        raise ValueError("no run of the job in sacct")
    last = instances[-1]
    if last.end is None or last.state in ("RUNNING", "PENDING", "REQUEUED"):
        raise ValueError(f"job {last.job} has not finished (state {last.state})")
    job = name or f"{last.name}-{last.job}"
    gpn = last.gpus // max(last.nnodes, 1)
    out: list[dict] = []

    def emit(t: datetime, event: str, **extra):
        if out:  # never go back in time: events from three clocks, rounded to the second
            t = max(t, parse_time(out[-1]["time"]))
        out.append({"time": iso(t), "job": job, "event": event, **extra})

    emit(instances[0].start, "start", nodes=last.nnodes, gpus_per_node=gpn)
    for n, inst in enumerate(instances):
        end = inst.end or last.end
        mine = [e for e in job_events if inst.start <= parse_time(e["time"]) <= end]
        if n > 0:
            emit(inst.start, "nodes_ready")
        running = next((parse_time(e["time"]) for e in mine if e["event"] == "running"), None)
        interrupted = inst.state == "REQUEUED" or n < len(instances) - 1
        rec = next((r for r in health if last.job in r.get("requeued", []) and
                    inst.start <= parse_time(r["time"]) <= end + SLACK), None) if interrupted else None
        fault_t, cause, node = end, "unknown", None
        if rec:
            f = rec["findings"][0]
            fault_t = parse_time(f["time"]) if f.get("time") else parse_time(rec["time"])
            fault_t = min(max(fault_t, inst.start), end)
            cause, node = cause_of(f), rec["node"]
        if n > 0:
            # a fault during startup still has to come after "running" for repo 12's state machine
            emit(min(running or fault_t, fault_t) if interrupted else (running or inst.start), "running")
        for e in mine:
            if e["event"] in ("checkpoint_start", "checkpoint_end") and (not interrupted or parse_time(e["time"]) <= fault_t):
                emit(parse_time(e["time"]), e["event"])
        if interrupted:
            emit(fault_t, "interrupt", cause=cause, **({"node": node} if node else {}))
            emit(parse_time(rec["time"]) if rec else end, "detected")
    emit(last.end, "end")
    return out


def summarize(instances: list[Instance]) -> str:
    rows = [f"{'run':>3}  {'state':<10} {'start':<20} {'end':<20} nodes"]
    for n, i in enumerate(instances):
        rows.append(f"{n:>3}  {i.state:<10} {iso(i.start):<20} {iso(i.end) if i.end else '-':<20} {i.nodes}")
    return "\n".join(rows)
