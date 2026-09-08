"""Refresh table artifacts after each newly completed three-seed group."""
import json
import time
import traceback
import importlib
import subprocess
import collect_tables
from status_report import update
from collect_tables import RUN

while True:
    try:
        state=RUN/'queue_state.json'
        current=json.loads(state.read_text(encoding='utf-8')) if state.exists() else {}
        if current.get('status')=='running' and 'benchmark' in (current.get('current_job') or ''):
            time.sleep(30);continue
        cache=collect_tables.HASH_CACHE
        importlib.reload(collect_tables)
        collect_tables.HASH_CACHE=cache
        changed=collect_tables.collect()
        update()
        if changed:
            subprocess.run(['C:/Users/evan/.conda/envs/ai4/python.exe',str(collect_tables.SUITE/'plot_baselines.py')],check=True)
        if state.exists() and json.loads(state.read_text(encoding='utf-8')).get('status') in ['complete','needs_attention']:
            break
    except Exception:
        traceback.print_exc()
    time.sleep(30)
