import json,pathlib,hashlib,datetime
root=pathlib.Path('/root/zhishitupui');b=root/'reproduction/main_tables_completion_20260907'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
state=json.loads((b/'QUEUE_STATE.json').read_text())
assert state['counts']=={'completed':39}
log={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'event':'before_final_three_seed_publication','purpose':'Confirm all39 jobs completed, publish final SS-AGA/E-PKG seeds17/29/43 and finish prior requested table/idea/data delivery','change':'CPU audit of saved outputs and document compilation only','seeds':[17,29,43],'command':'audit_main_completion.py; publish_available_main_tables.py --archive reproduction/sota/paper_support/document_versions/ssaga_epkg_three_seed_20260907_083100; validate_support_documents.py','process_gpu':'All six GPUs idle; scheduler queue finished; no new training or inference','results':'reproduction/language_table_seed17/main_completion_20260907/PUBLICATION_RESULTS.json','next':'Verify saved ranks and hashes, replace explicit two-seed interim row, compile PDFs, synchronize and stop'}
with (root/'SOTA_LOG.md').open('a') as f:f.write('\n'+json.dumps(log,ensure_ascii=False,indent=2)+'\n')
receipt={'timestamp':log['time'],'passed':False,'jobs':{},'source_paths':{},'plan_sha256':sha(b/'PLAN.json'),'queue_counts':state['counts']}
for jid,j in state['jobs'].items():
 assert j['status']=='completed' and j.get('exit_code',0)==0
 out=b/j['output'];c=json.loads((out/'config.json').read_text())
 for name,digest in c['source_hashes'].items():
  p=root/name.replace('\\','/')
  obj=b/'code_objects'/(digest+'.py')
  assert sha(p)==digest and sha(obj)==digest,(jid,name)
  receipt['source_paths'][name]={'sha256':digest,'object_path':str(obj.relative_to(root))}
 assert sha(pathlib.Path(j['return_archive']))==j['return_sha256']
 receipt['jobs'][jid]={'config_sha256':sha(out/'config.json'),'result_sha256':sha(out/'result.json'),'source_hashes':c['source_hashes'],'return_archive_sha256':j['return_sha256'],'state':j}
receipt['passed']=True
dest=b/'PUBLICATION_SOURCE_RECEIPT_FINAL.json'
dest.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':True,'jobs':len(receipt['jobs']),'source_paths':len(receipt['source_paths']),'receipt_path':str(dest),'receipt_sha256':sha(dest)}))

