"""Explicit repetition policy for the current, user-selected experiment batch."""
import math
import numpy as np

DEFAULT_SEEDS=[17,29,43]


def required_seeds(manifest,method,dataset,condition='full'):
    batch=(manifest or {}).get('method_batch',{})
    if (batch.get('repetition_policy')=='single_run_first' and condition=='full'
            and method in batch.get('methods',[]) and dataset in batch.get('datasets',[])):
        seeds=batch['seeds']
        assert seeds==[17], 'The single-run stage uses the preselected seed 17'
        assert f'{method.lower()}_{dataset}_s17' in batch['job_ids']
        return list(seeds)
    return list(DEFAULT_SEEDS)


def summarize(values):
    assert values and all(math.isfinite(v) for v in values)
    samples=np.array(values,dtype=np.float64)
    return float(samples.mean()),float(samples.std(ddof=1)) if len(samples)>1 else None
