"""gpuops: GPU operations for a Slurm cluster (lab).

    gpuops health                  the HealthCheckProgram (slurmd runs it on each node)
    gpuops sim inject|repair|replace|show   the lab's stand-in for DCGM
    gpuops status                  nodes, Slurm state and reason, GPU faults
    gpuops validate NODE           burn-in a repaired node inside a maintenance reservation, then resume it
    gpuops train / burnin          the lab's jobs
    gpuops events JOBID            sacct + job log + health log -> repo 12 event log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, events, health, policy, train, validate
from .slurm import Slurm, SlurmError
from .sources import NvidiaSmiSource, SimSource

SHARED = Path(os.environ.get("GPUOPS_SHARED", "/shared"))


def source_for(name: str, gpus: int):
    return NvidiaSmiSource() if name == "nvidia-smi" else SimSource(gpus_per_node=gpus)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gpuops", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("--gpus-per-node", type=int, default=int(os.environ.get("GPUOPS_GPUS_PER_NODE", "4")))
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("health", help="health check for this node (HealthCheckProgram)")
    h.add_argument("--node", default=None, help="default: $SLURMD_NODENAME or the host name")
    h.add_argument("--source", choices=("sim", "nvidia-smi"), default="sim")
    h.add_argument("--log", default=None, help="JSONL decision log (default: <shared>/gpu-health/<node>.jsonl)")
    h.add_argument("--temp-cordon", type=float, default=policy.TEMP_CORDON_C)
    h.add_argument("--dry-run", action="store_true", help="decide and print, change nothing")

    s = sub.add_parser("sim", help="the lab's simulated GPU telemetry")
    ss = s.add_subparsers(dest="sim_cmd", required=True)
    i = ss.add_parser("inject")
    i.add_argument("node")
    i.add_argument("--gpu", type=int, default=0)
    i.add_argument("--xid", type=int)
    i.add_argument("--dbe", type=int, default=0)
    i.add_argument("--row-remap-failure", action="store_true")
    i.add_argument("--temp", type=float)
    for name in ("repair", "replace"):
        ss.add_parser(name).add_argument("node")
    ss.add_parser("show").add_argument("node", nargs="?")

    sub.add_parser("status", help="nodes with Slurm state, reason and GPU findings")

    v = sub.add_parser("validate", help="burn-in a repaired node, then put it back in service")
    v.add_argument("node")
    v.add_argument("--burnin-seconds", type=int, default=15)
    v.add_argument("--force", action="store_true", help="also for a quarantined node (after a hardware replacement)")

    t = sub.add_parser("train", help="the lab training job (run it through jobs/train.sbatch)")
    t.add_argument("--steps", type=int, default=150)
    t.add_argument("--step-seconds", type=float, default=1.0)
    t.add_argument("--checkpoint-every", type=int, default=25)
    t.add_argument("--checkpoint-seconds", type=float, default=3.0)
    t.add_argument("--startup-seconds", type=float, default=5.0)
    t.add_argument("--run-dir", default=str(SHARED / "runs"))

    b = sub.add_parser("burnin", help="the lab burn-in job used by validate")
    b.add_argument("--gpus", type=int, required=True)
    b.add_argument("--seconds", type=int, default=15)

    e = sub.add_parser("events", help="build the repo 12 event log of a finished job")
    e.add_argument("job")
    e.add_argument("--sacct-file", help="saved `sacct -D -X -P -n -o ...` output instead of running sacct")
    e.add_argument("--run-dir", default=str(SHARED / "runs"))
    e.add_argument("--health-dir", default=str(SHARED / "gpu-health"))
    e.add_argument("--name", help="job name in the event log (default: <JobName>-<JobID>)")
    e.add_argument("-o", "--output", default="-")

    a = p.parse_args(argv)
    try:
        return _dispatch(a)
    except (SlurmError, ValueError, OSError) as err:
        print(f"gpuops: {err}", file=sys.stderr)
        return 2


def _dispatch(a) -> int:
    g = a.gpus_per_node
    if a.cmd == "health":
        node = a.node or health.node_name()
        log = Path(a.log) if a.log else SHARED / "gpu-health" / f"{node}.jsonl"
        rec = health.run(node, source_for(a.source, g), Slurm(dry_run=a.dry_run), None if a.dry_run else log, a.temp_cordon)
        if a.dry_run or rec["action"] != "none":
            print(json.dumps(rec))
        return 0
    if a.cmd == "sim":
        sim = SimSource(gpus_per_node=g)
        if a.sim_cmd == "inject":
            sim.inject(a.node, a.gpu, xid=a.xid, dbe=a.dbe, row_remap_failure=a.row_remap_failure, temp_c=a.temp)
        elif a.sim_cmd == "repair":
            sim.repair(a.node)
        elif a.sim_cmd == "replace":
            sim.replace(a.node)
        else:
            nodes = [a.node] if a.node else sorted(p.stem for p in sim.dir.glob("*.json"))
            for n in nodes:
                print(json.dumps({"node": n, "gpus": sim.read(n)}))
        return 0
    if a.cmd == "status":
        sim = SimSource(gpus_per_node=g)
        print(f"{'node':<8} {'state':<10} {'gpu policy':<11} reason")
        for n in Slurm().nodes():
            f = policy.evaluate(sim.read(n.name))
            print(f"{n.name:<8} {n.state:<10} {policy.worst(f):<11} {n.reason}")
        return 0
    if a.cmd == "validate":
        return validate.validate(a.node, Slurm(), g, a.burnin_seconds, a.force)
    if a.cmd == "train":
        return train.train(a.steps, a.step_seconds, a.checkpoint_every, a.checkpoint_seconds, a.startup_seconds, Path(a.run_dir))
    if a.cmd == "burnin":
        return validate.burnin(a.gpus, a.seconds, SimSource(gpus_per_node=g), health.node_name())
    if a.cmd == "events":
        text = Path(a.sacct_file).read_text() if a.sacct_file else events.run_sacct(a.job)
        inst = events.parse_sacct(text)
        evs = events.build(inst, events.read_jsonl(Path(a.run_dir) / a.job / "events.jsonl"),
                           events.health_records(Path(a.health_dir)), a.name)
        print(events.summarize(inst), file=sys.stderr)
        lines = "".join(json.dumps(x) + "\n" for x in evs)
        if a.output == "-":
            sys.stdout.write(lines)
        else:
            Path(a.output).write_text(lines)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
