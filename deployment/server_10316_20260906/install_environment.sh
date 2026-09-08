#!/usr/bin/env bash
set -euo pipefail
PROJECT=/root/zhishitupui
ENV_PREFIX="$PROJECT/.envs/kgc"
export CONDA_PKGS_DIRS="$PROJECT/.cache/conda/pkgs"
export PIP_CACHE_DIR="$PROJECT/.cache/pip"
export PIP_CONFIG_FILE="$PROJECT/deployment/pip.conf"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$PROJECT/deployment" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"
stage=python
trap 'rc=$?; printf "failed stage=%s exit=%s\n" "$stage" "$rc" > "$PROJECT/deployment/install.status"; exit "$rc"' ERR
cat > "$PIP_CONFIG_FILE" <<'EOF'
[global]
index-url = https://mirrors.sustech.edu.cn/pypi/web/simple
timeout = 60
retries = 3
EOF
printf 'running stage=%s\n' "$stage" > "$PROJECT/deployment/install.status"
if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    /home/vipuser/miniconda3/bin/conda create --prefix "$ENV_PREFIX" --override-channels -c conda-forge python=3.10.19 pip -y
fi
cp "$PIP_CONFIG_FILE" "$ENV_PREFIX/pip.conf"
PY="$ENV_PREFIX/bin/python"
stage=pip
printf 'running stage=%s\n' "$stage" > "$PROJECT/deployment/install.status"
"$PY" -m pip install --upgrade pip
stage=torch
printf 'running stage=%s\n' "$stage" > "$PROJECT/deployment/install.status"
"$PY" -m pip install 'torch==2.10.0+cu128' --index-url https://download.pytorch.org/whl/cu128
stage=python_dependencies
printf 'running stage=%s\n' "$stage" > "$PROJECT/deployment/install.status"
"$PY" -m pip install 'numpy==2.2.6' 'scipy==1.15.3' 'scikit-learn==1.7.2' 'pandas==2.3.3' 'matplotlib==3.10.8' 'psutil==7.2.2' 'tqdm==4.67.3' 'networkx==3.4.2' 'transformers==4.41.2' 'tokenizers==0.19.1' 'huggingface-hub==0.23.4' 'safetensors==0.7.0' 'tensorboard==2.20.0' 'torch-geometric==2.7.0'
stage=pyg_cuda_extensions
printf 'running stage=%s\n' "$stage" > "$PROJECT/deployment/install.status"
PYG_SOURCE=https://data.pyg.org/whl/torch-2.10.0+cu128.html
if [ -f "$PROJECT/deployment/wheelhouse/wheel_manifest.json" ]; then
    "$PY" - <<'PY'
import hashlib, json
from pathlib import Path
root=Path('/root/zhishitupui/deployment/wheelhouse')
manifest=json.loads((root/'wheel_manifest.json').read_text())
for row in manifest['wheels']:
    path=root/row['file']
    assert path.stat().st_size == row['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
print('All four uploaded official PyG wheels match their SHA-256 checksums.')
PY
    PYG_SOURCE="$PROJECT/deployment/wheelhouse"
fi
"$PY" -m pip install --no-index --only-binary=:all: --find-links "$PYG_SOURCE" 'pyg-lib==0.8.0+pt210cu128' 'torch-scatter==2.1.2+pt210cu128' 'torch-sparse==0.6.18+pt210cu128' 'torch-cluster==1.6.3+pt210cu128'
stage=dependency_check
printf 'running stage=%s\n' "$stage" > "$PROJECT/deployment/install.status"
"$PY" -m pip check
"$PY" -m pip freeze > "$PROJECT/deployment/requirements.lock.txt"
/home/vipuser/miniconda3/bin/conda list --prefix "$ENV_PREFIX" --explicit > "$PROJECT/deployment/conda-explicit.txt"
cat > "$PROJECT/activate.sh" <<'EOF'
#!/usr/bin/env bash
export KGC_ROOT=/root/zhishitupui
export PATH="$KGC_ROOT/.envs/kgc/bin:$PATH"
export PIP_CONFIG_FILE="$KGC_ROOT/deployment/pip.conf"
export PIP_CACHE_DIR="$KGC_ROOT/.cache/pip"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
EOF
printf 'completed\n' > "$PROJECT/deployment/install.status"
date -Is
