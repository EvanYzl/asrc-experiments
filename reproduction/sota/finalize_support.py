"""Stop the bounded support campaign after document delivery and full hash audit."""
import argparse
import datetime
import json
import subprocess
from frozen_data import ROOT,atomic_json,sha256
p=argparse.ArgumentParser();p.add_argument('mode',choices=['server','local']);mode=p.parse_args().mode
base=ROOT/'reproduction/sota';phase=base/'paper_support';receiptpath=phase/'SERVER_FINAL_RECEIPT.json'
def verify(items):
    for item in items:
        path=ROOT/item['path'];assert path.exists() and path.stat().st_size==item['bytes'] and sha256(path)==item['sha256'],item['path']
if mode=='server':
    from run_sota_queue import append_log
    assert not receiptpath.exists(),'Final receipt already exists; inspect and verify instead of finalizing twice'
    assert json.loads((phase/'EXPERIMENTS_COMPLETED.json').read_text())['status']=='completed'
    assert json.loads((phase/'DOCUMENT_VALIDATION.json').read_text())['support_final'] is True
    result=json.loads((phase/'RESULTS.json').read_text());assert result['status']=='completed'
    freeze=json.loads((phase/'EVALUATION_FREEZE.json').read_text());plan=json.loads((phase/'PLAN.json').read_text())
    docs=json.loads((ROOT/'deployment/paper_support/documents_server_receipt.json').read_text());verify(docs['files'])
    verified={}
    for mapping in [freeze['source_hashes'],freeze['input_hashes'],result['raw_hashes']]:
        for path,digest in mapping.items():assert sha256(ROOT/path)==digest,path;verified[path]=digest
    for ident,e in freeze['models'].items():
        path=e['training_path']+'/best.pt';assert sha256(ROOT/path)==e['checkpoint_sha256'],ident;verified[path]=e['checkpoint_sha256']
    assert sha256(ROOT/'outputs/kbs/_main/_tables/tables/t02.tex')==plan['table2_source_sha256']
    processes=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True).splitlines();workers=[]
    for line in processes:
        parts=line.split(None,2)
        if len(parts)<3 or not parts[1].startswith('python'):continue
        if any(('reproduction/sota/'+x) in parts[2] for x in ['train_complex.py','train_support.py','evaluate_support.py','profile_support.py','run_sota_queue.py','run_support_training.py','run_support_final_stages.py']):workers.append(line)
    assert not workers,workers
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    state={'phase':'paper_support','status':'completed_awaiting_user','seeds':[17,29,43],'table2_locked':True,'new_training':34,
           'new_once_only_test_runs':36,'reused_locked_test_runs':12,'isolated_profiles':21,'support_cells_filled':174,
           'external_baselines_started':0,'further_experiments_authorized':False,'timestamp':stamp}
    atomic_json(base/'TASK_SCOPE.json',state)
    text='# ASRC 当前状态：支持表格完成，等待用户指令\n\nTable 2 的四数据集三种子配置与结果保持冻结，仍仅达到暂定参考门槛。Table 4–8 已按预注册方案完成并回填 174 个单元格，Table 1–3 既有源文件与所有单元数据不变。\n\n最佳主方案：DBP-5L/DWY/WK3l 共享，E-PKG 独立，rank256/N3.01/Adagrad.1；不以新消融测试结果重新选主模型。\n\n关键负结果：E-PKG 全共享 MRR 比独立低 0.48 个百分点；DBP 去互逆增强均值反而高 0.30 个百分点，不能宣称互逆必要；对齐减半/10% 合成端点替换分别使共享 MRR 下降 11.71/7.82 个百分点。平均迁移收益不等于逐查询安全。所有结果与原始秩均保留。\n\n已完成：34 新训练，14 历史训练复用；36 新单次测试，12 锁定测试复用；21 个无并发 GPU 的验证查询计时；8 个实现检查不进入论文。文件：paper_support/RESULTS.json、EVALUATION_FREEZE.json、DOCUMENT_VALIDATION.json；idea 与四页主表 PDF 已更新并编译。\n\n当前无实验在运行，不再追加搜索、复现或优化。待办仅为用户下一步指令；外部 Table 2/3 缺项仍属于已有独立基线任务，不在本轮启动。恢复时先读 SOTA_LOG.md、TASK_SCOPE.json 及本记录。\n'
    (base/'PLAN.md').write_text(text,encoding='utf-8');(phase/'CURRENT_STATE.zh.md').write_text(text,encoding='utf-8')
    append_log('## Supporting-table campaign stopped — '+stamp+'\nPurpose: close the user-authorized Tables4–8 study and idea update.\nChanges/results: 174 supporting cells filled with registered three-seed results; 34 new trainings, 36 once-only tests, 21 isolated profiling runs. Existing T1–3 data and Table2 method configuration/results unchanged. All negative findings retained, including no-reciprocal +0.30 pp and E-PKG sharing −0.48 pp.\nSeeds/config: 17/29/43 and EVALUATION_FREEZE.json; GPU/PID: no campaign workers remain. No additional experiment scheduled.\nCommand: python reproduction/sota/finalize_support.py server. Outputs: main.tex → KBS_Main_Text_Tables.pdf; idea.std.zh source/PDF; design/tracker; raw ranks/timings/checkpoints; SERVER_FINAL_RECEIPT.json.\nVerification: all input/source/checkpoint/raw hashes and the delivered document bundle match. Conclusion: supporting experiment scope completed; no new fully comparable SOTA claim. Next: return immutable final snapshot, verify local hashes and wait for user instruction.')
    for item in docs['files']:verified[item['path']]=sha256(ROOT/item['path'])
    for path in [ROOT/'SOTA_LOG.md',base/'TASK_SCOPE.json',base/'PLAN.md',phase/'CURRENT_STATE.zh.md',phase/'RESULTS.json',phase/'EVALUATION_FREEZE.json',phase/'EXPERIMENTS_COMPLETED.json']:
        verified[path.relative_to(ROOT).as_posix()]=sha256(path)
    items=[{'path':path,'sha256':digest,'bytes':(ROOT/path).stat().st_size} for path,digest in sorted(verified.items())]
    atomic_json(receiptpath,{'status':'server_verified_no_experiments_running','timestamp':stamp,'files':items,'campaign_workers':workers,
        'gpu_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
        'summary':state})
    print(json.dumps({'server_verified_files':len(items),'checkpoints':48,'support_cells':174,'status':state['status']}))
else:
    receipt=json.loads(receiptpath.read_text());verify(receipt['files'])
    result={'status':'both_hosts_verified','timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'server_receipt_sha256':sha256(receiptpath),'verified_files':len(receipt['files']),'checkpoints':48,'support_cells':174,'experiments_running':False}
    atomic_json(phase/'LOCAL_FINAL_VERIFICATION.json',result);print(json.dumps(result))
