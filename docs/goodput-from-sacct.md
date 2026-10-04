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

CI run 11 (two GPU faults on a 2-node, 8-GPU job of 150 one-second steps, checkpoint every 25 steps). These are the real files from that run, kept in [tests/fixtures/ci-run](../tests/fixtures/ci-run); a unit test rebuilds the event log from them and checks it is identical.

```
sacct -D -X -P -n -o JobIDRaw,JobName,State,Start,End,NNodes,NodeList,AllocTRES,ExitCode
3|lab-train|REQUEUED|2026-10-04T17:05:17|2026-10-04T17:05:53|2|gpu-[1-2]|billing=8,cpu=8,gres/gpu=8,node=2|0:0
3|lab-train|REQUEUED|2026-10-04T17:06:53|2026-10-04T17:07:33|2|gpu-[3-4]|billing=8,cpu=8,gres/gpu=8,node=2|0:0
3|lab-train|COMPLETED|2026-10-04T17:08:53|2026-10-04T17:10:47|2|gpu-[1,4]|billing=8,cpu=8,gres/gpu=8,node=2|0:0
```

| | Fault 1 | Fault 2 |
|---|---|---|
| What | XID 79 on gpu-2, GPU 2 | XID 64 + row-remap failure on gpu-3, GPU 1 |
| Fault reported | 17:05:51 | 17:07:27 |
| Health check drains and requeues | 17:05:53 (+2 s) | 17:07:33 (+6 s) |
| Job starts again (other nodes) | 17:06:53 (+60 s) | 17:08:53 (+80 s) |
| Training resumes from checkpoint | 17:06:58 (+5 s) | 17:08:58 (+5 s) |

`python -m goodput analyze` (repo 12) on the event log, 330 s of 8 GPUs:

| Category | Share | Seconds |
|---|---|---|
| Productive | 47.0% | 155 |
| Checkpoint writes | 4.5% | 15 |
| Lost work (since the last checkpoint) | 0.6% | 2 |
| Detection (fault to requeue) | 2.4% | 8 |
| Waiting for nodes | 42.4% | 140 |
| Restart (startup until running) | 3.0% | 10 |

Two lessons, both visible only because the numbers come from a real scheduler:

- **Waiting dominated, with idle nodes available.** A requeued batch job cannot start until the launch credential of its previous run has expired (`AuthInfo=cred_expire`: 120 s by default, 30 s in this lab), and then it waits for the next scheduling pass. Here that was 60 to 80 s per requeue. On a two-minute lab job it is almost half the run; on a multi-week training job it is small per interruption, but it is paid on every one, by every GPU of the job.
- **Detection is the health-check interval.** The job hung instead of failing; nothing noticed until the next health check (2 and 6 s here, up to 10 s with this lab's interval). With a 5-minute interval, every hardware fault would cost up to 5 minutes of the whole job's GPUs.

The faults were injected right after a checkpoint, so almost no work was lost. Lost work grows with the time since the last checkpoint; that trade-off is what repo 12's Young/Daly calculation is about.
