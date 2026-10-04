import json
import tempfile
import unittest
from pathlib import Path

from gpuops import health
from tests.fakes import FakeSlurm, FakeSource, gpus_with


def run(slurm, gpus, log=None):
    return health.run("gpu-1", FakeSource(gpu_1=gpus), slurm, log, 90)


class Health(unittest.TestCase):
    def test_healthy_node_untouched(self):
        s = FakeSlurm(jobs={"gpu-1": ["7"]})
        rec = run(s, gpus_with(xid=(0, 13)))
        self.assertEqual(rec["action"], "none")
        self.assertEqual(s.calls, [])

    def test_drain_and_requeue(self):
        s = FakeSlurm(jobs={"gpu-1": ["7", "8"]})
        rec = run(s, gpus_with(xid=(1, 79)))
        self.assertEqual(s.calls, [("drain", "gpu-1", "gpu-health:drain XID 79 on GPU 1 (GPU has fallen off the bus)"),
                                   ("requeue", "7"), ("requeue", "8")])
        self.assertEqual(rec["requeued"], ["7", "8"])

    def test_cordon_does_not_requeue(self):
        s = FakeSlurm(jobs={"gpu-1": ["7"]})
        run(s, gpus_with(temp_c=(0, 95.0)))
        self.assertEqual([c[0] for c in s.calls], ["drain"])
        self.assertTrue(s.node("gpu-1").reason.startswith("gpu-health:cordon temperature on GPU 0"))

    def test_idempotent_and_escalates(self):
        s = FakeSlurm()
        run(s, gpus_with(xid=(1, 79)))
        run(s, gpus_with(xid=(1, 79)))
        self.assertEqual(len(s.calls), 1)  # the second check finds the node already drained for it
        g = gpus_with(xid=(1, 79))
        g[2]["xids"].append({"xid": 64})
        run(s, g)
        self.assertTrue(s.node("gpu-1").reason.startswith("gpu-health:quarantine XID 64"))

    def test_admin_reason_kept_but_jobs_requeued(self):
        s = FakeSlurm(jobs={"gpu-1": ["7"]})
        n = s.node("gpu-1")
        n.state, n.reason = "draining", "maintenance window"
        rec = run(s, gpus_with(xid=(0, 48)))
        self.assertEqual(s.calls, [("requeue", "7")])
        self.assertIn("someone else", rec["note"])
        self.assertEqual(n.reason, "maintenance window")

    def test_requeue_refused_is_recorded_not_fatal(self):
        s = FakeSlurm(jobs={"gpu-1": ["7"]}, not_requeueable={"7"})
        rec = run(s, gpus_with(xid=(0, 79)))
        self.assertEqual(rec["requeued"], [])
        self.assertIn("presently disabled", rec["errors"][0])

    def test_log_only_changes_and_new_findings(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "gpu-1.jsonl"
            s = FakeSlurm()
            run(s, gpus_with(), log)
            self.assertFalse(log.exists())
            for _ in range(3):
                run(s, gpus_with(xid=(2, 94)), log)  # watch: logged once
            for _ in range(3):
                run(s, gpus_with(xid=(2, 79)), log)  # drain: logged when it drains
            recs = [json.loads(l) for l in log.read_text().splitlines()]
            self.assertEqual([r["action"] for r in recs], ["watch", "drain"])
            self.assertEqual(recs[1]["findings"][0]["time"], "2026-10-04T10:00:00Z")

    def test_reason_action(self):
        self.assertEqual(health.reason_action("gpu-health:quarantine XID 64 on GPU 1"), "quarantine")
        self.assertIsNone(health.reason_action("maintenance"))
        self.assertIsNone(health.reason_action("gpu-health:bogus"))


if __name__ == "__main__":
    unittest.main()
