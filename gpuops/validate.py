"""Put a repaired node back into service, but only after it proves it works.

    1. refuse a quarantined node (the reason says the hardware has to be replaced)
    2. reserve the node for maintenance (Flags=MAINT), so nothing else lands on it, and resume it
    3. run the burn-in job on it inside the reservation (all its GPUs)
    4. pass: delete the reservation, the node takes jobs again
       fail: drain it again with the burn-in result as the reason

The same sequence works on a real cluster with a real burn-in in place of `gpuops burnin`.
"""

from __future__ import annotations

import os
import time

from . import health, policy
from .slurm import Slurm


def wait(slurm: Slurm, job: str, node: str, timeout_s: float, sleep=time.sleep) -> int:
    """Exit code of the burn-in job. If the health check drains the node again before the job
    starts (the fault is still there), the job would wait forever: cancel it and fail."""
    waited = 0.0
    while waited < timeout_s:
        state, rc = slurm.job(job)
        if rc is not None:
            return rc if state == "COMPLETED" or rc else 1
        if state == "PENDING" and slurm.node(node).drained:
            slurm.cancel(job)
            return 1
        sleep(2)
        waited += 2
    slurm.cancel(job)
    return 1


def validate(node: str, slurm: Slurm, gpus: int, burnin_seconds: int, force: bool = False, log=print,
             sleep=time.sleep) -> int:
    info = slurm.node(node)
    ours = health.reason_action(info.reason)
    if ours == "quarantine" and not force:
        log(f"{node}: quarantined ({info.reason}); replace the hardware, then run with --force")
        return 3
    if not info.drained:
        log(f"{node}: not drained ({info.state}); nothing to validate")
        return 0
    resv = f"validate-{node}"
    slurm.delete_reservation(resv)  # a leftover from an interrupted run
    slurm.create_reservation(resv, node, minutes=30)
    try:
        slurm.resume(node)
        log(f"{node}: resumed inside reservation {resv}; running burn-in ({burnin_seconds} s)")
        job = slurm.submit(
            f"--reservation={resv}", f"--nodelist={node}", "--nodes=1", f"--gres=gpu:{gpus}", "--exclusive",
            "--job-name=gpu-validate", "--no-requeue", f"--output={os.environ.get('GPUOPS_RUNS', '/shared/runs')}/validate-{node}-%j.out",
            f"--wrap=python3 -m gpuops burnin --gpus {gpus} --seconds {burnin_seconds}")
        rc = wait(slurm, job, node, timeout_s=burnin_seconds + 120, sleep=sleep)
        if rc == 0 and not slurm.node(node).drained:
            log(f"{node}: burn-in passed (job {job}); back in service")
            return 0
        reason = slurm.node(node).reason
        if not health.reason_action(reason):
            slurm.drain(node, f"{health.PREFIX}drain validation failed (burn-in job {job}, exit {rc})")
        log(f"{node}: validation failed (job {job}, exit {rc}); drained: {slurm.node(node).reason}")
        return 1
    finally:
        slurm.delete_reservation(resv)


def burnin(gpus: int, seconds: int, source, node: str, sleep=time.sleep) -> int:
    """The lab burn-in: every GPU allocated, and no fault reported while it runs."""
    visible = [g for g in (os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES", "")).split(",") if g]
    if len(visible) != gpus:
        print(f"burn-in: {len(visible)} GPU(s) allocated, {gpus} expected")
        return 1
    for _ in range(max(1, seconds)):
        bad = [f for f in policy.evaluate(source.read(node)) if policy.RANK[f.action] >= policy.RANK["cordon"]]
        if bad:
            print(f"burn-in: {bad[0].text()}")
            return 1
        sleep(1)
    print(f"burn-in: {gpus} GPUs, {seconds} s, no fault")
    return 0
