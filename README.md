# Slurm GPU Operations

GPU node operations for a Slurm cluster: a health check that drains nodes by XID policy and requeues the jobs a GPU fault has broken, a validation flow that only puts repaired nodes back, and goodput measured from `sacct`. All of it runs in Docker Compose and is tested end to end on every push.

**GPU fault → HealthCheckProgram (XID policy) → drain + requeue → job resumes from its checkpoint elsewhere → repair → burn-in in a maintenance reservation → back in service**

![ci](https://github.com/inigogonzalezgarcia/14-slurm-gpu-ops/actions/workflows/ci.yml/badge.svg)

> A learning-in-public lab about day 2 operations for GPU clusters. I don't have production GPU fleet or Slurm administration experience; this project is how I am learning the problem. The Slurm cluster is real (Slurm 23.11 from Ubuntu packages, with slurmdbd and MariaDB). The GPUs are not: each node has four placeholder devices and a JSON file that stands in for DCGM.

Sixth in a series: [09 – node remediation](https://github.com/inigogonzalezgarcia/09-gpu-node-remediation) (the same loop on Kubernetes, and the XID policy reused here), [10 – fleet observability](https://github.com/inigogonzalezgarcia/10-gpu-fleet-observability), [11 – fleet lifecycle](https://github.com/inigogonzalezgarcia/11-gpu-fleet-lifecycle), [12 – goodput and MTBI](https://github.com/inigogonzalezgarcia/12-gpu-goodput-mtbi) (the analyzer fed from `sacct` here), [13 – cluster acceptance](https://github.com/inigogonzalezgarcia/13-cluster-acceptance-burnin).

## What it does

| Piece | What it does |
|---|---|
| **Cluster** ([compose.yaml](compose.yaml)) | MariaDB, slurmdbd, slurmctld and 4 nodes with `Gres=gpu:lab:4`. Jobs get GPUs and `CUDA_VISIBLE_DEVICES` as on real hardware |
| **Health check** (`gpuops health`, Slurm `HealthCheckProgram`) | Every 10 s on each node: XID policy of repo 09 → `cordon` (drain, jobs finish), `drain` (drain + requeue jobs), `quarantine` (same, and validation refuses it). Never resumes a node; keeps a human's drain reason |
| **Requeue** | A job on a broken GPU hangs rather than crashes. The health check requeues it; Slurm restarts it on healthy nodes and it continues from its last checkpoint on shared storage |
| **Validation** (`gpuops validate NODE`) | Maintenance reservation on the node → resume → burn-in job on all its GPUs → release, or drain again with the result |
| **Goodput** (`gpuops events JOBID`) | `sacct -D` + the job's checkpoint log + the health-check log → the event log of repo 12 → `python -m goodput analyze` |

| Finding | Action | Running jobs |
|---|---|---|
| XID 13, 31, 43, 45 (application) | none | untouched |
| XID 94, unknown XIDs | watch (logged) | untouched |
| ≥ 90 °C | cordon | finish |
| XID 48, 63, 74, 79, 95, 119, 120, uncorrectable ECC | drain | requeued |
| XID 64, 92, row-remap failure | quarantine | requeued |

## CI run

Every push builds the cluster and runs [tests/e2e.sh](tests/e2e.sh): 36 checks in 8 scenarios, about 7 minutes. From run 11:

| Scenario | Result |
|---|---|
| GPUs | 4 nodes with `gpu:lab:4`; a job asking for 4 GPUs gets `CUDA_VISIBLE_DEVICES=0,1,2,3` |
| Application XID 13; XID 94 (contained memory error) | No drain; XID 94 logged as `watch` |
| Fault on a node an admin drained | Admin's reason kept |
| Training job, XID 79 on one of its nodes | Node drained 2 s after the fault, job requeued, restarted on two other nodes from its checkpoint |
| Same job, XID 64 + row-remap failure | Node quarantined, job requeued again, finished all 150 steps |
| `sacct -D` | Three records for the job: `REQUEUED`, `REQUEUED`, `COMPLETED` |
| Goodput (repo 12 analyzer) | 47.0%; waiting for the restart 42.4%, detection 2.4%, lost work 0.6% ([why](docs/goodput-from-sacct.md#what-a-lab-run-shows)) |
| Validation | Quarantined node refused; unrepaired node fails burn-in and is drained again; repaired node back in service; reset does not fix a row-remap failure, a replaced board does |
| Too hot (93 °C) | Cordoned; still drained after it cools down until validated |

## Run it

Docker with Compose. Python 3.11+ for the unit tests (standard library only).

```bash
docker compose up -d --build
docker compose exec slurmctld sinfo -N -o "%N %T %G"       # 4 nodes, gpu:lab:4 each
docker compose exec slurmctld sbatch /opt/gpuops/jobs/train.sbatch
docker compose exec slurmctld squeue                       # say it runs on gpu-[1-2]
docker compose exec slurmctld gpuops sim inject gpu-2 --gpu 1 --xid 79
docker compose exec slurmctld sinfo -R                     # gpu-2 drained by gpu-health, job requeued
git clone https://github.com/inigogonzalezgarcia/12-gpu-goodput-mtbi goodput-repo
GOODPUT_REPO=goodput-repo bash tests/e2e.sh            # the whole CI scenario
python -m unittest -v
docker compose down -v
```

## Documentation

- [docs/design.md](docs/design.md): the lab, how the GPUs are simulated, Kubernetes (repo 09) and Slurm side by side, the requeue delay
- [docs/runbook.md](docs/runbook.md): what to do when gpu-health drains a node
- [docs/goodput-from-sacct.md](docs/goodput-from-sacct.md): building repo 12's event log from Slurm, and what a real run shows
- [docs/decisions.md](docs/decisions.md): design decisions, lab shortcuts, what is missing

## Not tested here

- Real GPUs: the `nvidia-smi` + kernel-log source in `gpuops/sources.py` is written and unit-tested on sample text only.
- cgroup GPU isolation, NVML autodetection, Slurm versions other than 23.11.
- Running the policy as an NHC (Node Health Check) module, which is what most clusters use.

## Customisation and contact

Want to talk about Slurm, GPU node health or a lab like this for your team? Get in touch:

- Email: [inigogonzalezgarcia@yahoo.es](mailto:inigogonzalezgarcia@yahoo.es)
- LinkedIn: [linkedin.com/in/igonzalez93](https://www.linkedin.com/in/igonzalez93)

## License

MIT
