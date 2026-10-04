"""A stand-in for a training job, run by jobs/train.sbatch.

It does fixed-length "steps", writes a checkpoint every N steps to shared storage and, after a
requeue, resumes from the last checkpoint. If a GPU of one of its nodes reports a hardware fault it
stops making progress without exiting, as a real job stuck in a collective would: something outside
the job (here the health check) has to notice and requeue it.

Everything it does goes to <run-dir>/<job id>/events.jsonl, the training-framework side of the
goodput calculation (Slurm accounting is the other side).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from . import policy
from .sources import SimSource, now


def hostnames(nodelist: str) -> list[str]:
    """Expand a Slurm node list: 'gpu-[1,3-4],login' -> ['gpu-1', 'gpu-3', 'gpu-4', 'login']."""
    out = []
    for part in re.findall(r"[^,\[]+(?:\[[^\]]*\])?[^,]*", nodelist):
        m = re.fullmatch(r"(.*)\[([^\]]+)\](.*)", part)
        if not m:
            out.append(part)
            continue
        pre, ranges, post = m.groups()
        for r in ranges.split(","):
            a, _, b = r.partition("-")
            for i in range(int(a), int(b or a) + 1):
                out.append(f"{pre}{str(i).zfill(len(a))}{post}")
    return out


class Run:
    def __init__(self, directory: Path):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **extra) -> None:
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps({"time": now(), "event": event, **extra}) + "\n")

    def load(self) -> int:
        p = self.dir / "checkpoint.json"
        return json.loads(p.read_text())["step"] if p.exists() else 0

    def save(self, step: int) -> None:
        tmp = self.dir / "checkpoint.json.tmp"
        tmp.write_text(json.dumps({"step": step}))
        os.replace(tmp, self.dir / "checkpoint.json")


def broken_gpu(nodes: list[str], source, since: str) -> str | None:
    """A fault that would break this job: a drain or quarantine finding newer than the job's start."""
    for n in nodes:
        for f in policy.evaluate(source.read(n)):
            if policy.RANK[f.action] >= policy.RANK["drain"] and (f.time is None or f.time >= since):
                return f"{n}: {f.text()}"
    return None


def train(steps: int, step_s: float, every: int, ckpt_s: float, startup_s: float, run_dir: Path,
          source=None, sleep=time.sleep) -> int:
    job = os.environ.get("SLURM_JOB_ID", "local")
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    nodes = hostnames(os.environ.get("SLURM_JOB_NODELIST", "localhost"))
    source = source or SimSource()
    run = Run(run_dir / job)
    started = now()
    run.log("startup", restart=restart, nodes=nodes, gpus=os.environ.get("SLURM_JOB_GPUS", ""))
    sleep(startup_s)  # load the checkpoint, build the model, warm up the collectives
    step = run.load()
    run.log("running", step=step, restart=restart)
    stuck = None
    while step < steps:
        problem = broken_gpu(nodes, source, started)
        if problem:
            if stuck is None:
                stuck = problem
                print(f"{now()} no progress: {problem}", flush=True)  # the job's own log, not events
            sleep(1)
            continue
        sleep(step_s)
        step += 1
        if step % every == 0 and step < steps:
            run.log("checkpoint_start", step=step)
            sleep(ckpt_s)
            run.save(step)
            run.log("checkpoint_end", step=step)
    run.save(step)
    run.log("done", step=step, restart=restart)
    return 0
