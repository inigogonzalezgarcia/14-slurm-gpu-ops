import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from gpuops import sources
from gpuops.cli import main


class Sim(unittest.TestCase):
    def test_inject_repair_replace(self):
        with tempfile.TemporaryDirectory() as d:
            s = sources.SimSource(Path(d))
            self.assertEqual(len(s.read("gpu-1")), 4)
            s.inject("gpu-1", 2, xid=79)
            s.inject("gpu-1", 1, row_remap_failure=True, dbe=1, temp_c=95)
            g = s.read("gpu-1")
            self.assertEqual(g[2]["xids"][0]["xid"], 79)
            self.assertEqual((g[1]["dbe"], g[1]["row_remap_failure"], g[1]["temp_c"]), (1, True, 95))
            s.repair("gpu-1")
            g = s.read("gpu-1")
            self.assertEqual((g[2]["xids"], g[1]["dbe"], g[1]["temp_c"]), ([], 0, 41.0))
            self.assertTrue(g[1]["row_remap_failure"])  # a reset does not fix the memory
            s.replace("gpu-1")
            self.assertFalse(s.read("gpu-1")[1]["row_remap_failure"])
            self.assertEqual([p.name for p in Path(d).iterdir()], ["gpu-1.json"])  # no temp files left

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(sources.SimSource.__init__, "__defaults__", (Path(d), 4)):
                self.assertEqual(main(["sim", "inject", "gpu-3", "--gpu", "1", "--xid", "48"]), 0)
            self.assertEqual(json.loads((Path(d) / "gpu-3.json").read_text())["gpus"][1]["xids"][0]["xid"], 48)


class RealGPU(unittest.TestCase):
    """Parsers for the nvidia-smi source. Hand-written samples; never run against a real GPU here."""

    def test_nvidia_smi(self):
        g = sources.parse_nvidia_smi("0, 00000000:3B:00.0, 45, 0, No\n1, 00000000:5E:00.0, 91, 2, Yes\n2, 00000000:86:00.0, 40, [N/A], [N/A]\n")
        self.assertEqual([(x["index"], x["temp_c"], x["dbe"], x["row_remap_failure"]) for x in g],
                         [(0, 45.0, 0, False), (1, 91.0, 2, True), (2, 40.0, 0, False)])

    def test_kernel_xids(self):
        log = ("NVRM: Xid (PCI:0000:3b:00): 79, pid=1234, GPU has fallen off the bus.\n"
               "NVRM: Xid (PCI:0000:5e:00): 13, Graphics Exception\n")
        self.assertEqual(sources.parse_kernel_xids(log, {"00000000:3B:00.0": 0, "00000000:5E:00.0": 1}), {0: [79], 1: [13]})


if __name__ == "__main__":
    unittest.main()
