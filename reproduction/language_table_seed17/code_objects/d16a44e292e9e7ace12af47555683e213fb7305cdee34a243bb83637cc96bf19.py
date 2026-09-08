"""Atomic round-boundary checkpoints for long local baseline runs."""

import os
import random

import numpy as np
import torch


def save_round_checkpoint(path, model, optimizer, scheduler, completed_round,
                          ordered_domains, best_mrr, best_epoch, best_result):
    if not path:
        return
    checkpoint_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    temporary_path = checkpoint_path + '.tmp'
    state = {
        'format': 'multidomain_full_state_v1',
        'completed_round': int(completed_round),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'ordered_domains': list(ordered_domains),
        'best_mrr': float(best_mrr),
        'best_epoch': int(best_epoch),
        'best_result': best_result,
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(state, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def load_round_checkpoint(path, model, optimizer, scheduler, current_domains, device):
    if not path or not os.path.exists(path):
        return None
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('format') != 'multidomain_full_state_v1':
        raise ValueError(f'unsupported full-state checkpoint: {path}')
    restored_domains = list(checkpoint['ordered_domains'])
    if set(restored_domains) != set(current_domains):
        raise ValueError('resume checkpoint domain set does not match current dataset')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    random.setstate(checkpoint['python_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
    if torch.cuda.is_available() and checkpoint.get('cuda_rng_state_all') is not None:
        torch.cuda.set_rng_state_all([
            state.cpu() for state in checkpoint['cuda_rng_state_all']
        ])
    return {
        'start_round': int(checkpoint['completed_round']) + 1,
        'ordered_domains': restored_domains,
        'best_mrr': float(checkpoint['best_mrr']),
        'best_epoch': int(checkpoint['best_epoch']),
        'best_result': checkpoint['best_result'],
    }
