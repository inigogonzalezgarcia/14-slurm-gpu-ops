import unittest

from gpuops import policy
from tests.fakes import gpus_with


class Policy(unittest.TestCase):
    def test_same_table_as_repo_09(self):
        self.assertEqual(policy.HARDWARE_XIDS, [48, 63, 64, 74, 79, 92, 95, 119, 120])
        self.assertEqual(policy.xid_action(13)[0], "none")
        self.assertEqual(policy.xid_action(94)[0], "watch")
        self.assertEqual(policy.xid_action(999), ("watch", "not in the policy table"))

    def test_worst_finding_wins_and_comes_first(self):
        g = gpus_with(xid=(0, 13), temp_c=(1, 93.0))
        g[3]["xids"].append({"xid": 64, "time": None})
        f = policy.evaluate(g)
        self.assertEqual([x.action for x in f], ["quarantine", "cordon", "none"])
        self.assertEqual(policy.worst(f), "quarantine")
        self.assertEqual(f[0].text(), "XID 64 on GPU 3 (row remapping failure)")

    def test_counters(self):
        self.assertEqual(policy.worst(policy.evaluate(gpus_with(dbe=(2, 1)))), "drain")
        self.assertEqual(policy.worst(policy.evaluate(gpus_with(row_remap_failure=(2, True)))), "quarantine")
        self.assertEqual(policy.worst(policy.evaluate(gpus_with(temp_c=(2, 89.0)))), "none")
        self.assertEqual(policy.worst(policy.evaluate(gpus_with(temp_c=(2, 89.0)), temp_cordon_c=85)), "cordon")
        self.assertEqual(policy.worst(policy.evaluate(gpus_with())), "none")


if __name__ == "__main__":
    unittest.main()
