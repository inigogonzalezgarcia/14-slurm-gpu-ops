import json
import unittest
from pathlib import Path

from gpuops import events

FIX = Path(__file__).parent / "fixtures" / "ci-run"  # recorded from CI run 11 (see docs/goodput-from-sacct.md)


def load():
    inst = events.parse_sacct((FIX / "sacct.txt").read_text())
    return inst, events.read_jsonl(FIX / "job-events.jsonl"), events.read_jsonl(FIX / "gpu-health.jsonl")


class Sacct(unittest.TestCase):
    def test_parse(self):
        inst, _, _ = load()
        self.assertEqual([i.state for i in inst], ["REQUEUED", "REQUEUED", "COMPLETED"])
        self.assertEqual((inst[0].nnodes, inst[0].gpus, inst[0].nodes), (2, 8, "gpu-[1-2]"))

    def test_bad_line_and_never_started(self):
        with self.assertRaises(ValueError):
            events.parse_sacct("1|x|COMPLETED\n")
        line = "5|x|CANCELLED by 0|Unknown|2026-10-04T10:00:00|2|None assigned||0:0"
        self.assertEqual(events.parse_sacct(line), [])


class Build(unittest.TestCase):
    """Two GPU faults in the CI cluster: XID 79 on gpu-2, then XID 64 on gpu-3."""

    def test_same_event_log_as_the_ci_run(self):
        ev = events.build(*load())
        self.assertEqual(ev, events.read_jsonl(FIX / "events.jsonl"))
        by = [e for e in ev if e["event"] in ("interrupt", "detected", "nodes_ready", "running")]
        self.assertEqual([(e["event"], e["time"][11:19], e.get("cause")) for e in by], [
            ("interrupt", "17:05:51", "gpu_xid79"), ("detected", "17:05:53", None),
            ("nodes_ready", "17:06:53", None), ("running", "17:06:58", None),
            ("interrupt", "17:07:27", "gpu_xid64"), ("detected", "17:07:33", None),
            ("nodes_ready", "17:08:53", None), ("running", "17:08:58", None)])
        times = [e["time"] for e in ev]
        self.assertEqual(times, sorted(times))

    def test_unknown_cause_without_health_record(self):
        inst, job, _ = load()
        ev = events.build(inst, job, [])
        it = next(e for e in ev if e["event"] == "interrupt")
        self.assertEqual((it["cause"], it["time"]), ("unknown", "2026-10-04T17:05:53Z"))

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
