"""Exercise validation numbering against the actual round-checkpoint format."""
import tempfile
import unittest
from pathlib import Path

import torch

from common import ROOT
from run_graph_baseline import load_validation_progress


class GraphResumeProgressTests(unittest.TestCase):
    def setUp(self):
        self.parent=(ROOT/'reproduction/tmp').resolve()
        self.parent.mkdir(exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='graph_resume_',dir=self.parent)
        self.folder=Path(self.temporary.name).resolve()
        assert self.folder.is_relative_to(self.parent)

    def tearDown(self):
        assert self.folder.is_relative_to(self.parent) and self.folder!=self.parent
        self.temporary.cleanup()

    def best(self,index,score=.42):
        torch.save({'validation_mrr':score,'evaluation_index':index},self.folder/'best.pt')

    def last(self,completed_round):
        torch.save({'completed_round':completed_round},self.folder/'last.pt')

    def test_new_run_starts_at_first_evaluation(self):
        best,completed=load_validation_progress(self.folder)
        self.assertEqual((best,completed+1),(-1.,1))

    def test_early_best_does_not_rewind_the_training_counter(self):
        self.best(3)
        self.last(9)
        best,completed=load_validation_progress(self.folder)
        self.assertEqual(best,.42)
        self.assertEqual(completed+1,11)

    def test_interruption_between_best_and_last_replays_same_round(self):
        self.best(5)
        self.last(3)
        best,completed=load_validation_progress(self.folder)
        self.assertEqual((best,completed+1),(.42,5))

    def test_first_round_interruption_can_replay_from_initial_state(self):
        self.best(1)
        self.assertEqual(load_validation_progress(self.folder),(.42,0))

    def test_mismatched_checkpoint_pair_is_rejected(self):
        self.best(8)
        self.last(3)
        with self.assertRaises(AssertionError):
            load_validation_progress(self.folder)

    def test_missing_best_checkpoint_is_rejected(self):
        self.last(3)
        with self.assertRaises(AssertionError):
            load_validation_progress(self.folder)


if __name__=='__main__':
    unittest.main()
