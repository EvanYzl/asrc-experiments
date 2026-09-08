"""Round-boundary full-state checkpoint regression test for LSMGA."""

import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sources" / "LSMGA"
sys.path.insert(0, str(SOURCE))

from run_model import load_resume_checkpoint, save_resume_checkpoint  # noqa: E402


def training_step(model, optimizer):
    optimizer.zero_grad()
    inputs = torch.randn(7, 5)
    target = torch.randn(7, 3)
    loss = torch.square(model(inputs) - target).mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


def main():
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    model = torch.nn.Linear(5, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    training_step(model, optimizer)
    languages = ["fr", "el", "ja", "en", "es"]

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = str(Path(directory) / "resume.ckpt")
        save_resume_checkpoint(checkpoint, model, optimizer, 7, languages)

        expected_random = random.random()
        expected_numpy = np.random.rand(4)
        expected_torch = torch.rand(4)
        expected_loss = training_step(model, optimizer)
        expected_parameters = copy_parameters(model)

        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        restored_model = torch.nn.Linear(5, 3)
        restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.01)
        start_round, restored_languages = load_resume_checkpoint(
            checkpoint, restored_model, restored_optimizer,
            ["el", "en", "es", "fr", "ja"], torch.device("cpu"),
        )

        assert start_round == 8
        assert restored_languages == languages
        assert random.random() == expected_random
        np.testing.assert_array_equal(np.random.rand(4), expected_numpy)
        assert torch.equal(torch.rand(4), expected_torch)
        actual_loss = training_step(restored_model, restored_optimizer)
        assert torch.equal(actual_loss, expected_loss)
        for actual, expected in zip(copy_parameters(restored_model), expected_parameters):
            assert torch.equal(actual, expected)

    print("lsmga_full_state_resume_exact=True")


def copy_parameters(model):
    return [parameter.detach().clone() for parameter in model.parameters()]


if __name__ == "__main__":
    main()
