"""A fake Slurm and a fake GPU source for the unit tests."""

from gpuops.slurm import NodeInfo, SlurmError
from gpuops.sources import healthy_gpu


class FakeSlurm:
    def __init__(self, nodes=None, jobs=None, not_requeueable=()):
        self.nodes_ = {n: NodeInfo(n, "idle", "") for n in (nodes or ["gpu-1"])}
        self.jobs = jobs or {}  # node -> [job ids]
        self.not_requeueable = set(not_requeueable)
        self.calls = []
        self.job_states = []  # what job() returns, one per call
        self.on_resume = None

    def node(self, name):
        return self.nodes_[name]

    def nodes(self):
        return list(self.nodes_.values())

    def running_jobs(self, node):
        return list(self.jobs.get(node, []))

    def drain(self, node, reason):
        self.calls.append(("drain", node, reason))
        n = self.nodes_[node]
        n.state = "draining" if self.jobs.get(node) else "drained"
        n.reason = reason

    def resume(self, node):
        self.calls.append(("resume", node))
        self.nodes_[node].state, self.nodes_[node].reason = "maint", ""
        if self.on_resume:
            self.on_resume(self)

    def requeue(self, job):
        self.calls.append(("requeue", job))
        if job in self.not_requeueable:
            raise SlurmError(f"scontrol requeue {job}: Requested operation is presently disabled")

    def create_reservation(self, name, node, minutes):
        self.calls.append(("reserve", name, node))

    def delete_reservation(self, name):
        self.calls.append(("unreserve", name))

    def submit(self, *args):
        self.calls.append(("submit",) + args)
        return "99"

    def job(self, job):
        return self.job_states.pop(0) if len(self.job_states) > 1 else self.job_states[0]

    def cancel(self, job):
        self.calls.append(("cancel", job))


class FakeSource:
    def __init__(self, gpus=4, **per_node):
        self.per_node = per_node
        self.gpus = gpus

    def read(self, node):
        return self.per_node.get(node.replace("-", "_")) or [healthy_gpu(i) for i in range(self.gpus)]


def gpus_with(n=4, **changes):
    """gpus_with(xid=(1, 79)) -> 4 healthy GPUs, GPU 1 with XID 79."""
    gs = [healthy_gpu(i) for i in range(n)]
    for key, value in changes.items():
        if key == "xid":
            i, x = value
            gs[i]["xids"].append({"xid": x, "time": "2026-10-04T10:00:00Z"})
        else:
            i, v = value
            gs[i][key] = v
    return gs
