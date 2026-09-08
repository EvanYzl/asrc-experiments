"""Apply the user's new sequential, seed-17-only goal before any candidate test."""
import datetime
import json
import shutil
from frozen_data import ROOT,atomic_json
base=ROOT/'reproduction/sota'
assert not list((base/'final').glob('**/TEST_OPENED.json')),'Inspect already-opened old test before proceeding'
history=base/'history/unused_multiseed_acceptance';history.mkdir(parents=True,exist_ok=True)
for p in [base/'FINAL_EVALUATION_FREEZE.json',base/'final_queue/manifest.json']:
    if p.exists():shutil.move(str(p),str(history/p.name))
scope={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'single_seed_sequential',
       'seed':17,'order':['dbp5l','depkg','dwy','wk3l'],'current_dataset':'dbp5l','locked_datasets':[],
       'no_baseline_training':True,'no_additional_seeds':True,'reuse_completed_pilots':True,
       'stop_rule':'At least 10/12 strict wins, every other deficit <=0.005. One test per dataset after validation freeze. Lock before transfer. Stop if locked results make the overall threshold impossible.'}
atomic_json(base/'TASK_SCOPE.json',scope)
with (ROOT/'SOTA_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## New user goal: sequential seed-17 SOTA — '+scope['timestamp']+'\n')
    f.write('Purpose: first solve DBP-5L, then E-PKG, DWY, WK3l; only one dataset under method search at a time.\n')
    f.write('Action: archived unused multi-seed evaluation freeze and 12-job manifest before any test was opened. All GPUs idle; completed seed29/43 validation work is historical only and excluded from this phase. No further baseline or extra-seed jobs.\n')
    f.write('Configuration: reuse completed seed17 pilots, same unified ASRC algorithm, per-dataset validation-selected sharing hyperparameter. DBP shared r256 N3=.01 selected epoch35 on val_select.\n')
    f.write('Processes/GPU: none active. Result paths: single_seed/<dataset>. Next: freeze DBP only, test once, lock, check feasibility, then transfer in stated order. Keep user reference lines and unknown comparator sources; no SOTA claim from validation.\n')
print(json.dumps(scope))
