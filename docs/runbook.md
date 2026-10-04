# Runbook: GPU node drained by gpu-health

Commands run on the controller (`docker compose exec slurmctld bash` in the lab).

## 1. Why is the node out?

```bash
sinfo -R                      # every drained node and its reason
gpuops status                 # Slurm state next to what the GPU policy says now
tail -n 5 /shared/gpu-health/gpu-2.jsonl   # what the health check saw and did, with times
```

The reason says what happened: `gpu-health:<action> <finding>`.

| Action | Meaning | Running jobs |
|---|---|---|
| `cordon` | Too hot (≥ 90 °C). Usually airflow or the room, not the GPU | Left to finish |
| `drain` | Hardware XID (48, 63, 74, 79, 95, 119, 120) or uncorrectable ECC. The GPU needs at least a reset | Requeued |
| `quarantine` | XID 64 / 92 or a row-remap failure. The memory cannot be repaired in place | Requeued |

A reason without the `gpu-health:` prefix was set by a person; the health check leaves it alone.

## 2. Check the jobs

```bash
sacct -D -X -j <job> -o JobID,State,Start,End,NodeList,Restarts
squeue -t PENDING -o "%A %j %r %S"   # requeued jobs wait ~cred_expire before they can start
```

A job submitted with `--no-requeue` is not requeued: the health log has an `errors` entry. Ask the owner, or cancel it.

## 3. Repair

| Action | Repair |
|---|---|
| `cordon` | Fix cooling; nothing to do on the node |
| `drain` | GPU reset (`nvidia-smi -r` with nothing running) or reboot; reseat if it repeats |
| `quarantine` | Hardware replacement (RMA) |

In the lab: `gpuops sim repair <node>` (reset) or `gpuops sim replace <node>` (new board).

## 4. Put it back

```bash
gpuops validate <node>            # quarantined nodes: gpuops validate <node> --force, after the replacement
```

It reserves the node (`validate-<node>`, `Flags=MAINT`), resumes it, runs the burn-in job on all its GPUs, and then either deletes the reservation (back in service) or drains it again with the result as the reason. Exit codes: 0 back in service, 1 failed, 3 quarantined (refused without `--force`).

Never `scontrol update NodeName=<node> State=RESUME` directly for a gpu-health drain: if the fault is still there the health check drains it again within one interval, after a job may already have landed on it.

## 5. Afterwards

If the same node comes back to this runbook within a few weeks, quarantine it and open an RMA, even if the XID was a "drain" one.
