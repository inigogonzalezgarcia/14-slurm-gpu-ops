"""Where GPU health comes from.

SimSource is what the lab uses: one JSON file per node in a shared directory, written by
`gpuops sim` (a stand-in for DCGM). NvidiaSmiSource is the shape of the real thing and is NOT
tested here (no GPU in this lab): nvidia-smi for ECC, row remapping and temperature, the kernel
log for XIDs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SIM_DIR = Path(os.environ.get("GPU_SIM_DIR", "/var/lib/gpu-sim"))


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def healthy_gpu(i: int) -> dict:
    return {"index": i, "xids": [], "dbe": 0, "row_remap_failure": False, "temp_c": 41.0}


class SimSource:
    def __init__(self, directory: Path = SIM_DIR, gpus_per_node: int = 4):
        self.dir = Path(directory)
        self.gpus_per_node = gpus_per_node

    def path(self, node: str) -> Path:
        return self.dir / f"{node}.json"

    def read(self, node: str) -> list[dict]:
        p = self.path(node)
        if not p.exists():
            return [healthy_gpu(i) for i in range(self.gpus_per_node)]
        return json.loads(p.read_text())["gpus"]

    def write(self, node: str, gpus: list[dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=f".{node}.")
        with os.fdopen(fd, "w") as f:
            json.dump({"node": node, "gpus": gpus}, f, indent=1)
        os.chmod(tmp, 0o644)
        os.replace(tmp, self.path(node))  # readers never see a half-written file

    def inject(self, node: str, gpu: int, xid: int | None = None, dbe: int = 0,
               row_remap_failure: bool = False, temp_c: float | None = None) -> None:
        gpus = self.read(node)
        g = gpus[gpu]
        if xid is not None:
            g["xids"].append({"xid": xid, "time": now()})
        g["dbe"] += dbe
        g["row_remap_failure"] = g["row_remap_failure"] or row_remap_failure
        if temp_c is not None:
            g["temp_c"] = temp_c
        self.write(node, gpus)

    def repair(self, node: str) -> None:
        """A GPU reset: clears XIDs, volatile ECC counts and temperature. A row remap failure stays."""
        gpus = self.read(node)
        for g in gpus:
            g.update(xids=[], dbe=0, temp_c=41.0)
        self.write(node, gpus)

    def replace(self, node: str) -> None:
        """The hardware vendor swapped the board: everything healthy again."""
        self.write(node, [healthy_gpu(i) for i in range(self.gpus_per_node)])


XID_RE = re.compile(r"NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\): (\d+),")


def _bus(pci: str) -> str:
    """'00000000:3B:00.0' (nvidia-smi) and '0000:3b:00' (kernel log) -> '3b:00'."""
    return ":".join(pci.lower().split(".")[0].split(":")[-2:])


def parse_kernel_xids(log: str, bus_to_index: dict[str, int]) -> dict[int, list[int]]:
    """XIDs per GPU index from kernel log lines like 'NVRM: Xid (PCI:0000:3b:00): 79, pid=...'.
    An XID whose bus matches no GPU is filed under index -1."""
    index = {_bus(b): i for b, i in bus_to_index.items()}
    out: dict[int, list[int]] = {}
    for m in XID_RE.finditer(log):
        out.setdefault(index.get(_bus(m.group(1)), -1), []).append(int(m.group(2)))
    return out


def parse_nvidia_smi(csv_text: str) -> list[dict]:
    """Rows of: index, pci.bus_id, temperature.gpu, ecc.errors.uncorrected.volatile.total, remapped_rows.failure"""
    gpus = []
    for line in csv_text.strip().splitlines():
        idx, bus, temp, dbe, remap = [c.strip() for c in line.split(",")]
        num = lambda v: 0 if v in ("[N/A]", "N/A", "[Not Supported]") else int(float(v))
        gpus.append({"index": int(idx), "bus": bus, "temp_c": float(num(temp)), "dbe": num(dbe),
                     "row_remap_failure": remap.lower() in ("yes", "1"), "xids": []})
    return gpus


class NvidiaSmiSource:
    """Untested in this lab. XIDs are read from the kernel log since boot, so they stay
    visible until the node reboots, like the sim's latched faults."""

    def read(self, node: str) -> list[dict]:
        q = "index,pci.bus_id,temperature.gpu,ecc.errors.uncorrected.volatile.total,remapped_rows.failure"
        smi = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=30).stdout
        gpus = parse_nvidia_smi(smi)
        log = subprocess.run(["journalctl", "-k", "-b", "--no-pager", "-o", "cat"],
                             capture_output=True, text=True, timeout=30).stdout
        xids = parse_kernel_xids(log, {g["bus"]: g["index"] for g in gpus})
        for g in gpus:
            g["xids"] = [{"xid": x} for x in xids.get(g["index"], [])]
        if -1 in xids and gpus:  # cannot tell which GPU: charge the node through its first GPU
            gpus[0]["xids"] += [{"xid": x} for x in xids[-1]]
        return gpus
