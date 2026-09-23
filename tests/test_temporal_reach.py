"""Date-order edge cases that static reachability cannot distinguish."""
import unittest

import pandas as pd

from aml.temporal_reach import temporal_seed_reach


def calculate(gids, seeds, transfers):
    nodes = pd.DataFrame({"gid": gids, "is_seed": [gid in seeds for gid in gids]})
    tx = pd.DataFrame(transfers, columns=["src", "dst", "date"])
    return temporal_seed_reach(nodes, tx).set_index("gid")


class TemporalReachTests(unittest.TestCase):
    def test_reverse_chronology_does_not_create_a_temporal_path(self):
        result = calculate([1, 2, 3, 4], {1}, [
            (1, 2, "2026-07-03"), (2, 3, "2026-07-01"), (3, 4, "2026-07-04"),
        ])
        self.assertEqual(result.loc[2].tolist(), [1, 1])
        self.assertEqual(result.loc[3].tolist(), [0, 0])
        self.assertEqual(result.loc[4].tolist(), [0, 0])

    def test_same_day_closure_is_an_upper_bound_independent_of_row_order(self):
        transfers = [(1, 2, "2026-07-01"), (2, 3, "2026-07-01"), (3, 4, "2026-07-01")]
        forward = calculate([1, 2, 3, 4], {1}, transfers)
        backward = calculate([1, 2, 3, 4], {1}, list(reversed(transfers)))
        pd.testing.assert_frame_equal(forward, backward)
        self.assertEqual(forward.loc[2].tolist(), [1, 1])
        self.assertEqual(forward.loc[3].tolist(), [1, 0])
        self.assertEqual(forward.loc[4].tolist(), [1, 0])

    def test_isolates_are_preserved_and_seed_never_counts_itself(self):
        result = calculate([7, 2, 1, 9], {1, 7}, [(1, 2, "2026-07-01")])
        self.assertEqual(list(result.index), [7, 2, 1, 9])
        self.assertEqual(result.loc[7].tolist(), [0, 0])
        self.assertEqual(result.loc[1].tolist(), [0, 0])
        self.assertEqual(result.loc[9].tolist(), [0, 0])

    def test_cycles_do_not_inflate_counts_but_other_seeds_can_count(self):
        result = calculate([1, 2, 3], {1, 3}, [
            (1, 2, "2026-07-01"), (2, 1, "2026-07-01"),
            (3, 2, "2026-07-01"), (2, 3, "2026-07-02"),
        ])
        self.assertEqual(result.loc[1].tolist(), [1, 0])
        self.assertEqual(result.loc[2].tolist(), [2, 2])
        self.assertEqual(result.loc[3].tolist(), [1, 1])

    def test_paths_can_wait_and_progress_only_on_later_dates(self):
        result = calculate([1, 2, 3, 4], {1}, [
            (1, 2, "2026-07-01"), (2, 3, "2026-07-05"), (3, 4, "2026-07-31"),
        ])
        self.assertTrue(result.loc[[2, 3, 4]].eq(1).all().all())

    def test_no_transactions_returns_zero_for_every_node(self):
        result = calculate([1, 2], {1}, [])
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(result.eq(0).all().all())


if __name__ == "__main__":
    unittest.main()
