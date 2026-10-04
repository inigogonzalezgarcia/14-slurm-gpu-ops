import json
import unittest
from pathlib import Path

from gpuops import events

FIX = Path(__file__).parent / "fixtures" / "run"


def load():
    inst = events.parse_sacct((FIX / "sacct.txt").read_text())
    return inst, events.read_jsonl(FIX / "runs" / "2" / "events.jsonl"), events.health_records(FIX / "gpu-health")


class Sacct(unittest.TestCase):
    def test_parse(self):
        inst, _, _ = load()
        self.assertEqual([i.state for i in inst], ["REQUEUED", "COMPLETED"])
        self.assertEqual((inst[0].nnodes, inst[0].gpus, inst[0].nodes), (2, 8, "gpu-[3-4]"))

    def test_bad_line_and_never_started(self):
        with self.assertRaises(ValueError):
            events.parse_sacct("1|x|COMPLETED\n")
        line = "5|x|CANCELLED by 0|Unknown|2026-10-04T10:00:00|2|None assigned||0:0"
        self.assertEqual(events.parse_sacct(line), [])


class Build(unittest.TestCase):
    """A real local run (docs/goodput-from-sacct.md): XID 48 on gpu-3 at 15:50:45, requeued at 15:50:48."""

    def test_events(self):
        ev = events.build(*load())
        self.assertEqual([e["event"] for e in ev], [
            "start", "checkpoint_start", "checkpoint_end", "interrupt", "detected", "nodes_ready", "running",
            "checkpoint_start", "checkpoint_end", "checkpoint_start", "checkpoint_end", "end"])
        by = {e["event"]: e for e in ev}
        self.assertEqual(by["start"], {**by["start"], "time": "2026-10-04T15:50:18Z", "nodes": 2, "gpus_per_node": 4})
        self.assertEqual(by["interrupt"], {**by["interrupt"], "time": "2026-10-04T15:50:45Z", "cause": "gpu_xid48", "node": "gpu-3"})
        self.assertEqual(by["detected"]["time"], "2026-10-04T15:50:48Z")
        self.assertEqual(by["nodes_ready"]["time"], "2026-10-04T15:53:19Z")
        self.assertEqual(by["running"]["time"], "2026-10-04T15:53:24Z")
        self.assertEqual(ev[-1]["time"], "2026-10-04T15:54:15Z")
        times = [e["time"] for e in ev]
        self.assertEqual(times, sorted(times))

    def test_unknown_cause_without_health_record(self):
        inst, job, _ = load()
        ev = events.build(inst, job, [])
        it = next(e for e in ev if e["event"] == "interrupt")
        self.assertEqual((it["cause"], it["time"]), ("unknown", "2026-10-04T15:50:48Z"))

    def test_unfinished_job(self):
        inst, job, h = load()
        inst[-1].state, inst[-1].end = "RUNNING", None
        with self.assertRaisesRegex(ValueError, "has not finished"):
            events.build(inst, job, h)

    def test_cause_names(self):
        self.assertEqual(events.cause_of({"what": "XID 79"}), "gpu_xid79")
        self.assertEqual(events.cause_of({"what": "ECC DBE"}), "gpu_ecc_dbe")


if __name__ == "__main__":
    unittest.main()
