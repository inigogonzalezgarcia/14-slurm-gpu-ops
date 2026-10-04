import contextlib
import io
import os
import unittest
from unittest import mock

from gpuops import validate
from tests.fakes import FakeSlurm, FakeSource, gpus_with


def drained(s, reason="gpu-health:drain XID 79 on GPU 1 (GPU has fallen off the bus)"):
    n = s.node("gpu-1")
    n.state, n.reason = "drained", reason
    return s


def run(s, **kw):
    msgs = []
    rc = validate.validate("gpu-1", s, 4, 5, log=msgs.append, sleep=lambda _: None, **kw)
    return rc, msgs


class Validate(unittest.TestCase):
    def test_pass(self):
        s = drained(FakeSlurm())
        s.job_states = [("PENDING", None), ("RUNNING", None), ("COMPLETED", 0)]
        rc, msgs = run(s)
        self.assertEqual(rc, 0)
        self.assertEqual([c[0] for c in s.calls], ["unreserve", "reserve", "resume", "submit", "unreserve"])
        submit = s.calls[3]
        self.assertIn("--reservation=validate-gpu-1", submit)
        self.assertIn("--gres=gpu:4", submit)
        self.assertIn("back in service", msgs[-1])

    def test_burnin_fails(self):
        s = drained(FakeSlurm())
        s.job_states = [("FAILED", 1)]
        rc, _ = run(s)
        self.assertEqual(rc, 1)
        self.assertEqual(s.node("gpu-1").reason, "gpu-health:drain validation failed (burn-in job 99, exit 1)")
        self.assertEqual(s.calls[-1][0], "unreserve")

    def test_fault_still_there_job_cancelled(self):
        s = drained(FakeSlurm())
        s.job_states = [("PENDING", None)]
        s.on_resume = lambda fs: drained(fs)  # the health check drains it again at once
        rc, _ = run(s)
        self.assertEqual(rc, 1)
        self.assertIn(("cancel", "99"), s.calls)

    def test_quarantine_needs_force(self):
        s = drained(FakeSlurm(), "gpu-health:quarantine XID 64 on GPU 1 (row remapping failure)")
        self.assertEqual(run(s)[0], 3)
        self.assertEqual(s.calls, [])
        s.job_states = [("COMPLETED", 0)]
        self.assertEqual(run(s, force=True)[0], 0)

    def test_not_drained(self):
        s = FakeSlurm()
        self.assertEqual(run(s)[0], 0)
        self.assertEqual(s.calls, [])


class Burnin(unittest.TestCase):
    def burn(self, env, gpus):
        with mock.patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(io.StringIO()) as out:
            rc = validate.burnin(4, 3, FakeSource(gpu_1=gpus), "gpu-1", sleep=lambda _: None)
        return rc, out.getvalue()

    def test_all_gpus_and_no_fault(self):
        self.assertEqual(self.burn({"SLURM_JOB_GPUS": "0,1,2,3"}, gpus_with())[0], 0)
        self.assertEqual(self.burn({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, gpus_with())[0], 0)
        rc, out = self.burn({"SLURM_JOB_GPUS": "0,1"}, gpus_with())
        self.assertEqual((rc, out.strip()), (1, "burn-in: 2 GPU(s) allocated, 4 expected"))
        rc, out = self.burn({"SLURM_JOB_GPUS": "0,1,2,3"}, gpus_with(row_remap_failure=(1, True)))
        self.assertEqual(rc, 1)
        self.assertIn("row remap failure on GPU 1", out)


if __name__ == "__main__":
    unittest.main()
