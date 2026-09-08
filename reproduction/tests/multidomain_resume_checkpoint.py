"""Full-state resume regression test shared by DMKGC and IMKGC."""

import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


SOURCE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SOURCE))

from src.checkpointing import load_round_checkpoint, save_round_checkpoint  # noqa: E402


def step(model, optimizer, scheduler):
    optimizer.zero_grad()
    loss = torch.square(model(torch.randn(7, 5)) - torch.randn(7, 3)).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach()


def parameters(model):
    return [parameter.detach().clone() for parameter in model.parameters()]


def main():
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    model = torch.nn.Linear(5, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=20)
    step(model, optimizer, scheduler)
    domains = ["fr", "el", "ja", "en", "es"]
    best_result = {"ja": [0.1, 0.2, torch.tensor(0.3)]}

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = str(Path(directory) / "resume.ckpt")
        save_round_checkpoint(
            checkpoint, model, optimizer, scheduler, 7, domains,
            0.3, 4, best_result,
        )
        expected_random = random.random()
        expected_numpy = np.random.rand(4)
        expected_torch = torch.rand(4)
        expected_loss = step(model, optimizer, scheduler)
        expected_parameters = parameters(model)
        expected_scheduler = scheduler.state_dict()

        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        restored_model = torch.nn.Linear(5, 3)
        restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.01)
        restored_scheduler = torch.optim.lr_scheduler.LinearLR(restored_optimizer, total_iters=20)
        state = load_round_checkpoint(
            checkpoint, restored_model, restored_optimizer, restored_scheduler,
            ["el", "en", "es", "fr", "ja"], torch.device("cpu"),
        )
        assert state["start_round"] == 8
        assert state["ordered_domains"] == domains
        assert state["best_mrr"] == 0.3
        assert state["best_epoch"] == 4
        assert torch.equal(state["best_result"]["ja"][2], torch.tensor(0.3))
        assert random.random() == expected_random
        np.testing.assert_array_equal(np.random.rand(4), expected_numpy)
        assert torch.equal(torch.rand(4), expected_torch)
        actual_loss = step(restored_model, restored_optimizer, restored_scheduler)
        assert torch.equal(actual_loss, expected_loss)
        assert restored_scheduler.state_dict() == expected_scheduler
        for actual, expected in zip(parameters(restored_model), expected_parameters):
            assert torch.equal(actual, expected)

    print(f"source={SOURCE.name}")
    print("multidomain_full_state_resume_exact=True")


if __name__ == "__main__":
    main()
