"""GPU health policy: the XID table of repo 09 (gpu-node-remediation), mapped to Slurm actions.

Actions, from least to most severe:

    none        nothing to do (application XIDs, healthy GPU)
    watch       record it; a human looks at it
    cordon      drain the node, let running jobs finish (too hot: the room, not the GPU)
    drain       drain the node and requeue its jobs (the GPU needs a reset before it is trusted)
    quarantine  drain the node and requeue its jobs; validation refuses it (hardware replacement)

The worst action across the GPUs of a node wins: Slurm schedules whole nodes for these jobs.
"""

from __future__ import annotations

from dataclasses import dataclass

ACTIONS = ("none", "watch", "cordon", "drain", "quarantine")
RANK = {a: i for i, a in enumerate(ACTIONS)}

# XID -> (action, short meaning). Same table as repo 09 docs/xid-policy.md; the actions are this
# lab's choice, derived from the causes NVIDIA documents for each XID, not an NVIDIA recommendation.
XID_POLICY: dict[int, tuple[str, str]] = {
    13: ("none", "graphics engine exception (usually the application)"),
    31: ("none", "GPU memory page fault (usually the application)"),
    43: ("none", "GPU stopped processing (application fault)"),
    45: ("none", "preemptive cleanup (follows another error)"),
    94: ("watch", "contained memory error"),
    48: ("drain", "double-bit ECC error"),
    63: ("drain", "row remapping pending"),
    74: ("drain", "NVLink error"),
    79: ("drain", "GPU has fallen off the bus"),
    95: ("drain", "uncontained memory error"),
    119: ("drain", "GSP RPC timeout"),
    120: ("drain", "GSP error"),
    64: ("quarantine", "row remapping failure"),
    92: ("quarantine", "high single-bit ECC error rate"),
}
HARDWARE_XIDS = sorted(x for x, (a, _) in XID_POLICY.items() if RANK[a] >= RANK["drain"])

TEMP_CORDON_C = 90


@dataclass(frozen=True)
class Finding:
    gpu: int
    action: str
    what: str  # "XID 79", "ECC DBE", "temperature"
    detail: str
    time: str | None = None  # when the GPU reported it, if known

    def text(self) -> str:
        return f"{self.what} on GPU {self.gpu} ({self.detail})"


def xid_action(xid: int) -> tuple[str, str]:
    return XID_POLICY.get(xid, ("watch", "not in the policy table"))


def evaluate(gpus: list[dict], temp_cordon_c: float = TEMP_CORDON_C) -> list[Finding]:
    """Findings for one node, worst first. Each GPU: {"index", "xids": [{"xid", "time"}], "dbe",
    "row_remap_failure", "temp_c"} (the shape of gpuops.sources)."""
    out: list[Finding] = []
    for g in gpus:
        i = int(g["index"])
        for x in g.get("xids", []):
            action, meaning = xid_action(int(x["xid"]))
            out.append(Finding(i, action, f"XID {int(x['xid'])}", meaning, x.get("time")))
        if g.get("dbe", 0) > 0:
            out.append(Finding(i, "drain", "ECC DBE", f"{g['dbe']} uncorrectable ECC error(s)"))
        if g.get("row_remap_failure"):
            out.append(Finding(i, "quarantine", "row remap failure", "memory cannot be repaired in place"))
        t = g.get("temp_c")
        if t is not None and t >= temp_cordon_c:
            out.append(Finding(i, "cordon", "temperature", f"{t:.0f} °C, limit {temp_cordon_c:.0f} °C"))
    out.sort(key=lambda f: (-RANK[f.action], f.gpu))
    return out


def worst(findings: list[Finding]) -> str:
    return findings[0].action if findings else "none"
