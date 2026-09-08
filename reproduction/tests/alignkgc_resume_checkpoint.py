"""Exact full-state resume regression test for AlignKGC."""

import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


SOURCE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SOURCE))

from AlignKGC.alignkgc_base import (  # noqa: E402
    AlignKgcBaseTrainer,
    _load_checkpoint,
)


def make_holder(checkpoint, save_directory):
    holder = object.__new__(AlignKgcBaseTrainer)
    holder.scoring_function = torch.nn.Linear(5, 3)
    holder.optim = torch.optim.Adagrad(
        holder.scoring_function.parameters(), lr=0.03)
    holder.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        holder.optim, mode="max", patience=2)
    holder.best_mrr_on_valid = {"valid_score": {"m": {"mrr": 0.4}}}
    holder.equiv_rel = {
        "1": {2: (torch.tensor(0.25), 8)},
        "2": {},
    }
    holder.b = torch.tensor(90.0, requires_grad=True)
    holder.resume_checkpoint = str(checkpoint)
    holder.save_directory = str(save_directory)
    holder._resume_payload = None
    return holder


def train_step(holder):
    holder.optim.zero_grad()
    inputs = torch.randn(7, 5)
    targets = torch.randn(7, 3)
    loss = torch.square(holder.scoring_function(inputs) - targets).mean()
    loss.backward()
    holder.optim.step()
    return loss.detach()


def parameters(holder):
    return [parameter.detach().clone()
            for parameter in holder.scoring_function.parameters()]


def main():
    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        checkpoint = directory / "resume.ckpt"
        original = make_holder(checkpoint, directory / "run")
        train_step(original)
        original.scheduler.step(0.4)
        original.b.data.fill_(87.5)
        original.save_resume_checkpoint(999)

        expected_python = random.random()
        expected_numpy = np.random.rand(4)
        expected_torch = torch.rand(4)
        expected_loss = train_step(original)
        expected_parameters = parameters(original)

        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        restored = make_holder(checkpoint, directory / "unused")
        restored._resume_payload = _load_checkpoint(checkpoint, "cpu")
        assert restored._resume_payload["save_directory"] == original.save_directory
        next_batch = restored.load_resume_checkpoint()
        assert next_batch == 1000
        assert restored.b.item() == 87.5
        assert restored.equiv_rel["1"][2][1] == 8
        assert restored.equiv_rel["1"][2][0].item() == 0.25
        assert random.random() == expected_python
        np.testing.assert_array_equal(np.random.rand(4), expected_numpy)
        assert torch.equal(torch.rand(4), expected_torch)
        actual_loss = train_step(restored)
        assert torch.equal(actual_loss, expected_loss)
        for actual, expected in zip(parameters(restored), expected_parameters):
            assert torch.equal(actual, expected)

    print("alignkgc_full_state_resume_exact=True")


if __name__ == "__main__":
    main()
