# Design

## The lab

```
 compose.yaml
 ┌──────────┐   ┌──────────┐   ┌───────────┐
 │ mariadb  │◀──│ slurmdbd │◀──│ slurmctld │  operator commands run here (sinfo, sbatch, gpuops)
 └──────────┘   └──────────┘   └─────┬─────┘
                                     │
         ┌───────────────┬───────────┴───┬───────────────┐
     ┌───┴───┐       ┌───┴───┐       ┌───┴───┐       ┌───┴───┐
     │ gpu-1 │       │ gpu-2 │       │ gpu-3 │       │ gpu-4 │   slurmd, gres gpu:lab:4,
     └───────┘       └───────┘       └───────┘       └───────┘   HealthCheckProgram every 10 s
  volumes shared by all: /var/lib/gpu-sim (simulated GPU telemetry), /shared (job output,
  checkpoints, health-check logs)
```

One image for every Slurm role, built from Ubuntu 24.04 packages (Slurm 23.11, MUNGE). MariaDB is the official image. Nothing else is downloaded.

**Simulated GPUs.** Each node declares `Gres=gpu:lab:4`. `gres.conf` points at `/dev/labgpu0..3`, character devices the entrypoint creates with the major/minor of `/dev/null`: Slurm checks that a GPU's `File=` is a device, and with no cgroup device constraint it never opens it. Slurm then allocates GPUs and sets `CUDA_VISIBLE_DEVICES` exactly as it would with real ones. What the lab cannot show is anything that needs a driver: NVML autodetection, cgroup device isolation, real XIDs.

**cgroups in a container.** Slurm 23.11's slurmd always loads a cgroup plugin, and on a cgroup v2 host it expects systemd to delegate a cgroup to it. There is no systemd here, so `cgroup.conf` sets `IgnoreSystemd=yes` and the GPU nodes run `privileged` with an init as PID 1 (`init: true`). Before slurmd starts, the entrypoint moves the container's processes out of the cgroup root and enables the controllers below it, which is what systemd would have done. Each of these was a CI failure first: slurmd refusing to start, then every job step failing to launch, then a doubled cgroup path because slurmd was PID 1.

**Simulated telemetry.** `/var/lib/gpu-sim/<node>.json` holds per-GPU XIDs (with the time they were reported), uncorrectable ECC count, row-remap failure and temperature. `gpuops sim inject|repair|replace` changes it; the health check and the jobs read it. It plays the role DCGM or the kernel log plays on a real node. A fault stays until `repair` (a GPU reset) or `replace` (new board); a row-remap failure survives a reset.

## The pieces

| Piece | Runs where | What it does |
|---|---|---|
| `gpuops health` (`bin/gpu-health`) | every node, by slurmd, as root | Policy → drain / requeue / log. Never resumes |
| `gpuops validate NODE` | controller, by an operator | Maintenance reservation → resume → burn-in job → release or drain again |
| `gpuops train` (`jobs/train.sbatch`) | a 2-node job | Steps, checkpoints, resume after requeue; hangs on a GPU fault |
| `gpuops events JOBID` | controller | `sacct -D` + job log + health log → repo 12 event log |
| `gpuops status` | controller | Node state and reason next to the GPU policy verdict |

## Kubernetes (repo 09) and Slurm, side by side

Repo 09 built the same loop as a Kubernetes controller. Most ideas map one to one; the differences are where the schedulers differ.

| Repo 09 (Kubernetes) | Here (Slurm) |
|---|---|
| Controller polls dcgm-exporter metrics | `HealthCheckProgram` on each node, every `HealthCheckInterval` |
| Cordon | `State=DRAIN` with nothing else (`cordon` action, too hot) |
| Cordon + evict pods (PDB respected) | `State=DRAIN` + `scontrol requeue` of the node's jobs (`drain`, `quarantine`) |
| State in node annotations | State in the node's `Reason` (`gpu-health:<action> …`) |
| Kubernetes Events | JSON Lines decision log per node |
| Validation Job pinned to the node | Burn-in job inside a `MAINT` reservation on the node |
| `max-unavailable` budget | Not built: see [decisions](decisions.md#what-is-missing) |
| Quarantined label, human removes it | `gpu-health:quarantine` reason, `validate --force` after replacement |

The biggest practical difference: Kubernetes recreates a pod somewhere else by itself, while a Slurm job that loses a GPU usually just hangs (a collective waits for a peer that will never answer) until something kills it. That is why the health check requeues, and why detection time is bounded by `HealthCheckInterval`.

## The job and its requeues

`jobs/train.sbatch` asks for 2 nodes with all 4 GPUs each, `--exclusive` and `--requeue`. The job checks its nodes' GPUs every step; a hardware fault reported after it started makes it stop progressing without exiting. The health check on the faulty node drains it and runs `scontrol requeue`. Slurm kills the job, puts it back in the queue with `Restarts=1` and, after the credential-expiry delay (below), starts it on two other nodes. The job reads `checkpoint.json` from shared storage and continues from the last checkpoint. `SLURM_JOB_ID` stays the same; `sacct -D` shows each run as its own record (`REQUEUED`, …, `COMPLETED`).

**The requeue delay.** A requeued batch job cannot start again until the launch credential of its previous run has expired: Slurm sets its begin time to now + `cred_expire` + 1 s. `cred_expire` defaults to 120 s; the lab sets `AuthInfo=cred_expire=30` to keep the run short. The job then starts at the next scheduling pass: 60 to 80 s after the requeue in the CI run. It shows up in the goodput numbers as "waiting for nodes", even with idle nodes available.
