#!/usr/bin/env bash
set -euo pipefail
PROJECT=/root/zhishitupui
DEPLOY="$PROJECT/deployment"
stage=waiting_for_installation
trap 'rc=$?; printf "failed stage=%s exit=%s\n" "$stage" "$rc" > "$DEPLOY/checks.status"; exit "$rc"' ERR
printf 'running stage=%s\n' "$stage" > "$DEPLOY/checks.status"
for attempt in $(seq 1 180); do
    status=$(cat "$DEPLOY/install.status")
    if [ "$status" = completed ]; then break; fi
    if [[ "$status" == failed* ]]; then
        printf '%s\n' "$status"
        false
    fi
    sleep 10
done
test "$(cat "$DEPLOY/install.status")" = completed
source "$PROJECT/activate.sh"
PY="$PROJECT/.envs/kgc/bin/python"
stage=dataset_validation
printf 'running stage=%s\n' "$stage" > "$DEPLOY/checks.status"
timeout --kill-after=10s 180s "$PY" "$DEPLOY/verify_data.py" > "$DEPLOY/data_verification.log" 2>&1
stage=six_gpu_validation
printf 'running stage=%s\n' "$stage" > "$DEPLOY/checks.status"
timeout --kill-after=10s 240s "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=6 "$DEPLOY/verify_environment.py" > "$DEPLOY/environment_verification.log" 2>&1
stage=library_imports
printf 'running stage=%s\n' "$stage" > "$DEPLOY/checks.status"
"$PY" - <<'PY'
import json, importlib, importlib.metadata, sys
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
root=Path('/root/zhishitupui/deployment')
modules=['numpy','scipy','sklearn','pandas','matplotlib','psutil','tensorboard','transformers','torch_geometric','torch_scatter','torch_sparse','torch_cluster','pyg_lib']
for name in modules:
    importlib.import_module(name)
with SummaryWriter(str(root/'tensorboard_smoke')) as writer:
    writer.add_scalar('environment_smoke/value',1.,0)
report={'passed':True,'python':sys.version,'imports':modules,'tensorboard_write_passed':True}
(root/'library_import_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
PY
"$PY" -m pip check > "$DEPLOY/pip_check.log" 2>&1
"$PY" - <<'PY'
import ast, json, os, subprocess, sys
from pathlib import Path
root=Path('/root/zhishitupui/deployment')
expected='https://mirrors.sustech.edu.cn/pypi/web/simple'
listing=subprocess.check_output([sys.executable,'-m','pip','config','list'],text=True)
matches=[ast.literal_eval(line.split('=',1)[1]) for line in listing.splitlines() if line.startswith('global.index-url=')]
assert matches == [expected], 'Effective pip index does not match the requested mirror'
(root/'pip_index.txt').write_text(expected+'\n')
(root/'pip_configuration_check.json').write_text(json.dumps({'passed':True,'index_url':expected,'config_file':os.environ['PIP_CONFIG_FILE'],'verification':'pip config list; custom PIP_CONFIG_FILE'},indent=2)+'\n')
PY
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader > "$DEPLOY/gpus_after_checks.csv"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader > "$DEPLOY/gpu_processes_after_checks.csv"
printf 'completed\n' > "$DEPLOY/checks.status"
date -Is
