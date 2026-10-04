# Decisions

## 1. A HealthCheckProgram, not a daemon

Slurm already runs a program on every node at a fixed interval, as root, and keeps running it while the node is drained. Using that hook means no extra service to deploy or monitor. Real clusters usually run LBNL Node Health Check (NHC) there; the policy in `gpuops/policy.py` could be called from an NHC check instead of being the whole program. The cost: the interval is the detection time for a hung job, and the program has 60 s before slurmd kills it.

## 2. The health check requeues jobs, not only drains

Draining alone lets running jobs finish, which is right for a node that is too hot and wrong for a GPU that has fallen off the bus: the job will not finish, it will hang. So `drain` and `quarantine` also requeue the node's running jobs. A job submitted with `--no-requeue` is left alone and the refusal is logged; deciding to cancel someone's job is a human call.

## 3. Never resume automatically

A fault that disappears from telemetry has not necessarily been fixed (a GPU reset clears the XID, not the cause). The health check only takes nodes out. A node comes back only through `gpuops validate`, which runs a burn-in on it first, inside a maintenance reservation so no user job lands on it in the meantime.

## 4. Respect a human's drain

If an admin drained a node, their reason stays: overwriting "RMA ticket 1234" with an XID would lose information. The health check still requeues the jobs a hardware fault has broken, and records that it saw the fault.

## 5. The reason field is the state

`gpu-health:<action> <finding>` in the node's `Reason` is all the state there is: `sinfo -R` shows why every node is out, `validate` reads it to refuse a quarantined node, and the health check reads it to avoid redraining or to escalate (drain → quarantine). No database to keep in sync.

## 6. Same XID policy as repo 09

The table in `gpuops/policy.py` is the one in [repo 09's xid-policy.md](https://github.com/inigogonzalezgarcia/09-gpu-node-remediation/blob/main/docs/xid-policy.md), built from the causes NVIDIA documents for each XID. The actions are this lab's choice, not an NVIDIA recommendation.

## 7. Goodput from three sources

Slurm accounting knows when each run started and ended but not why it ended, nor when checkpoints happened. The job knows its checkpoints. The health check knows the fault and when it acted. `gpuops events` joins the three into the event log of repo 12, so the same analyzer measures the simulator there and this cluster here.

## 8. Lab shortcuts

- Jobs run as root, and the MUNGE key is created in the image. Both are wrong for a real cluster.
- Telemetry is a shared JSON file per node instead of DCGM on each node.
- The GPU nodes run privileged, so slurmd can manage its own cgroups without systemd. No cgroup limits are applied (`proctrack/linuxproc`, `task/none`): GPUs are allocated but not isolated.
- Slurm 23.11 from the Ubuntu archive, not the latest release.

## What is missing

- **A disruption budget.** Repo 09 limits how many nodes can be in remediation at once; here every faulty node is drained. In Slurm this belongs in the health check (count nodes with a `gpu-health:` reason before draining) or in a controller-side script.
- **Repeat offenders.** A node drained twice in a short window should go to quarantine. The decision log has the data; the rule is not written.
- **NHC integration** and the `nvidia-smi` source, which are written but untested here (no GPU).
- **Slurm's own requeue on exit codes** (`RequeueExit`) for jobs that crash instead of hanging.
