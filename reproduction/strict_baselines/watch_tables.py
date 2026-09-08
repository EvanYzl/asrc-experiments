"""Refresh artifacts when the authorized single-run or multi-seed result is ready."""
import json
import time
import traceback
import importlib
import subprocess
import collect_tables
import status_report
import table_scope
from collect_tables import RUN

while True:
    try:
        state=RUN/'queue_state.json'
        current=json.loads(state.read_text(encoding='utf-8')) if state.exists() else {}
        if current.get('status')=='running' and 'benchmark' in (current.get('current_job') or ''):
            time.sleep(30);continue
        cache=collect_tables.HASH_CACHE
        importlib.reload(table_scope)
        importlib.reload(collect_tables)
        collect_tables.HASH_CACHE=cache
        changed=collect_tables.collect()
        importlib.reload(status_report)
        status_report.update()
        audit=json.loads((RUN/'results/table_fill_audit.json').read_text(encoding='utf-8'))
        figure_status=RUN/'figures/FIGURE_STATUS.json'
        figures=json.loads(figure_status.read_text(encoding='utf-8')) if figure_status.exists() else {}
        if changed or figures.get('published_groups',figures.get('three_seed_groups'))!=audit.get('published_groups',audit['three_seed_groups']):
            subprocess.run(['C:/Users/evan/.conda/envs/ai4/python.exe',str(collect_tables.SUITE/'plot_baselines.py')],check=True)
        if state.exists() and json.loads(state.read_text(encoding='utf-8')).get('status') in ['complete','needs_attention']:
            break
    except Exception:
        traceback.print_exc()
    time.sleep(30)
