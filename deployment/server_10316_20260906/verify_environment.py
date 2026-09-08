"""Small six-GPU math/PyG and NCCL checks; this does not train an experiment."""
from pathlib import Path
import datetime
import importlib.metadata as metadata
import json
import os
import sys
import torch
import torch.distributed as dist
import torch_scatter
import torch_sparse
import torch_cluster
import pyg_lib
from torch_geometric.nn import GCNConv
from transformers import AdamW, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
rank = int(os.environ.get('LOCAL_RANK', '0'))
world = int(os.environ.get('WORLD_SIZE', '1'))
assert world == 6 and torch.cuda.device_count() == 6
torch.set_num_threads(2)
torch.cuda.set_device(rank)
device = torch.device('cuda', rank)
torch.manual_seed(17 + rank)
x = torch.randn(128, 128, device=device, requires_grad=True)
y = (x @ x.T).square().mean()
y.backward()
assert torch.isfinite(y) and torch.isfinite(x.grad).all()
src = torch.tensor([1.,2.,3.,4.], device=device, requires_grad=True)
idx = torch.tensor([0,0,1,1], device=device)
reduced = torch_scatter.scatter_add(src, idx)
assert reduced.tolist() == [3.,7.]
reduced.sum().backward()
maximum, _ = torch_scatter.scatter_max(src.detach(), idx)
assert maximum.tolist() == [2.,4.]
edge = torch.tensor([[0,1,1,2],[1,0,2,1]], device=device)
gnn = GCNConv(8, 4).to(device)
features = torch.randn(3, 8, device=device, requires_grad=True)
out = gnn(features, edge)
out.square().mean().backward()
assert out.shape == (3,4) and torch.isfinite(out).all() and torch.isfinite(features.grad).all()
coalesced_edges, coalesced_values = torch_sparse.coalesce(edge, torch.ones(4,device=device), 3, 3)
assert coalesced_edges.shape == (2,4) and torch.isfinite(coalesced_values).all()
neighbors = torch_cluster.knn_graph(torch.randn(8,3,device=device), k=2)
assert neighbors.shape == (2,16)
with torch.autocast('cuda', dtype=torch.float16):
    half_result = torch.randn(32,32,device=device) @ torch.randn(32,32,device=device)
assert half_result.dtype == torch.float16 and torch.isfinite(half_result).all()
dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=90), device_id=device)
token = torch.tensor([rank + 1.], device=device)
dist.all_reduce(token)
assert token.item() == 21.
dist.barrier()
report = {'rank':rank, 'name':torch.cuda.get_device_name(rank),
          'capability':list(torch.cuda.get_device_capability(rank)),
          'total_memory':torch.cuda.get_device_properties(rank).total_memory,
          'bf16_native_supported':torch.cuda.is_bf16_supported(including_emulation=False),
          'fp16_passed':True, 'matmul_backward_passed':True,
          'scatter_passed':True, 'pyg_gcn_backward_passed':True,
          'sparse_and_cluster_passed':True, 'nccl_all_reduce':token.item()}
(ROOT / f'deployment/gpu_check_rank{rank}.json').write_text(json.dumps(report,indent=2) + '\n')
dist.barrier()
dist.destroy_process_group()
if rank == 0:
    versions = {name:metadata.version(name) for name in ['torch','torch-geometric','torch-scatter','torch-sparse','torch-cluster','pyg-lib','numpy','scipy','scikit-learn','pandas','matplotlib','psutil','transformers','tensorboard']}
    summary = {'passed':True, 'python':sys.version, 'executable':sys.executable,
               'versions':versions, 'cuda_runtime':torch.version.cuda,
               'cudnn':torch.backends.cudnn.version(), 'nccl':torch.cuda.nccl.version(),
               'cuda_arch_list':torch.cuda.get_arch_list(), 'gpu_count':6,
               'devices':[json.loads((ROOT / f'deployment/gpu_check_rank{i}.json').read_text()) for i in range(6)]}
    (ROOT / 'deployment/environment_verification.json').write_text(json.dumps(summary,indent=2) + '\n')
    print(json.dumps(summary))
