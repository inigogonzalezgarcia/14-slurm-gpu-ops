import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from gpuops import train
from tests.fakes import FakeSource, gpus_with


class Stop(Exception):
    pass


class Train(unittest.TestCase):
    def test_hostnames(self):
        self.assertEqual(train.hostnames("gpu-[1,3-4]"), ["gpu-1", "gpu-3", "gpu-4"])
        self.assertEqual(train.hostnames("gpu-2"), ["gpu-2"])
        self.assertEqual(train.hostnames("n[08-10],login"), ["n08", "n09", "n10", "login"])

    def run_job(self, d, env, source, max_sleeps=None):
        calls = []

        def sleep(s):
            calls.append(s)
            if max_sleeps and len(calls) > max_sleeps:
                raise Stop()

        with mock.patch.dict(os.environ, env, clear=False), contextlib.redirect_stdout(io.StringIO()):
            try:
                return train.train(10, 1, 4, 2, 5, Path(d), source=source, sleep=sleep)
            except Stop:
                return None

    def events(self, d):
        return [json.loads(l) for l in (Path(d) / "42" / "events.jsonl").read_text().splitlines()]

    def test_checkpoints_and_resume(self):
        env = {"SLURM_JOB_ID": "42", "SLURM_JOB_NODELIST": "gpu-[1-2]", "SLURM_RESTART_COUNT": "0"}
        with tempfile.TemporaryDirectory() as d:
            # a fault on gpu-2 after the job started: no progress, the job never exits by itself
            faulty = gpus_with(xid=(3, 79))
            faulty[3]["xids"][0]["time"] = "9999-01-01T00:00:00Z"
            src = FakeSource()
            sleeps = []
            orig = src.read

            def read(node):
                sleeps.append(node)
                return faulty if node == "gpu-2" and len(sleeps) > 12 else orig(node)

            src.read = read
            self.assertIsNone(self.run_job(d, env, src, max_sleeps=30))
            ev = self.events(d)
            self.assertEqual([e["event"] for e in ev], ["startup", "running", "checkpoint_start", "checkpoint_end"])
            self.assertEqual(json.loads((Path(d) / "42" / "checkpoint.json").read_text())["step"], 4)
            # the requeued run starts from step 4 and finishes
            env["SLURM_RESTART_COUNT"] = "1"
            self.assertEqual(self.run_job(d, env, FakeSource()), 0)
            ev = self.events(d)[4:]
            self.assertEqual(ev[1], {**ev[1], "event": "running", "step": 4, "restart": 1})
            self.assertEqual([e.get("step") for e in ev if e["event"] == "checkpoint_end"], [8])
            self.assertEqual(ev[-1]["event"], "done")
            self.assertEqual(ev[-1]["step"], 10)

    def test_old_faults_do_not_block(self):
        # a fault reported before the job started belongs to a node that should not have been
        # allocated; the job does not hang on it (the health check deals with the node)
        env = {"SLURM_JOB_ID": "42", "SLURM_JOB_NODELIST": "gpu-1"}
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.run_job(d, env, FakeSource(gpu_1=gpus_with(xid=(0, 79)))), 0)


if __name__ == "__main__":
    unittest.main()
