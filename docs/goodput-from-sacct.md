# Goodput from Slurm accounting

Repo 12 ([gpu-goodput-mtbi](https://github.com/inigogonzalezgarcia/12-gpu-goodput-mtbi)) measures where a training job's GPU-hours go, from an event log. Its docs list where each event could come from on a real cluster but did not test it. This repo does, for Slurm.

```bash
gpuops events <jobid> > events.jsonl          # on the controller
python -m goodput analyze events.jsonl        # in a checkout of repo 12
```

| Repo 12 event | Source here |
|---|---|
| `start` | `sacct -D`: start of the first run of the job |
| `checkpoint_start`, `checkpoint_end` | The job's own log (`/shared/runs/<job>/events.jsonl`) |
| `interrupt` | The health-check log: the time the GPU reported the fault, and its cause (`gpu_xid79`…) and node |
| `detected` | The health-check log: when it requeued the job |
| `nodes_ready` | `sacct -D`: start of the next run |
| `running` | The job's log: first `running` event of the next run (after loading the checkpoint) |
| `end` | `sacct -D`: end of the last run |

`sacct` is called with `-D` (every requeued run is its own record, otherwise only the last one is shown), `-X` (allocations, not steps), `-P` (parsable) and `SLURM_TIME_FORMAT=%Y-%m-%dT%H:%M:%S` with `TZ=UTC`. Times from the three sources are rounded to the second and never allowed to go backwards. A requeue with no matching health-check record (someone ran `scontrol requeue` by hand, or the job crashed) gets cause `unknown` at the end of the run.

## What a lab run shows

The unit tests use a real run from a local Slurm build (`tests/fixtures/run`), recorded before the lab's `cred_expire` was lowered to 30 s:

```
start 15:50:18 → checkpoint 15:50:38–41 → XID 48 on gpu-3 15:50:45 → requeued 15:50:48
→ next run starts 15:53:19 → running 15:53:24 → two checkpoints → end 15:54:15
```

| Category | Share |
|---|---|
| Productive | 27.4% |
| Checkpoint writes | 3.8% |
| Lost work (since the last checkpoint) | 1.7% |
| Detection (fault to requeue) | 1.3% |
| Waiting for nodes | 63.7% |
| Restart (startup until running) | 2.1% |

Two lessons, both visible only because the numbers come from a real scheduler:

- **Waiting dominated, with idle nodes available.** That is the requeue delay: a requeued job waits for its previous launch credential to expire (`AuthInfo=cred_expire`, 120 s by default) before it can start. On a two-minute lab job it is most of the run; on a multi-week training job it is small per interruption, but it is paid on every one. Shortening `cred_expire` shortens it.
- **Detection is the health-check interval.** The job hung instead of failing; nothing noticed until the next health check (3 s here, up to 10 s with this lab's interval). With a 5-minute interval, every hardware fault would cost up to 5 minutes of the whole job's GPUs.

The CI run with `cred_expire=30` and two faults is in the [README](../README.md#ci-run).
