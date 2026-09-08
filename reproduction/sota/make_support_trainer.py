"""Generate a minimally changed validation trainer; never alter the frozen original."""
from pathlib import Path
base=Path(__file__).parent
source=(base/'train_complex.py').read_text(encoding='utf-8')
source=source.replace('from frozen_data import ROOT,DOMAINS,atomic_json,sha256,load_multikg',
    'from frozen_data import ROOT,DOMAINS,atomic_json,sha256\nfrom support_data import build_inputs\nfrom types import SimpleNamespace')
start=source.index('    p=argparse.ArgumentParser();p.add_argument')
end=source.index("    assert not (out/'result.json')",start)
source=source[:start]+'''    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--smoke',action='store_true');cli=p.parse_args()
    entry=json.loads((ROOT/'reproduction/sota/paper_support/PLAN.json').read_text())['models'][cli.id]
    recipe=entry['recipe'].copy();assert entry['trainer']=='support' and not entry['reuse_training']
    if cli.smoke:recipe.update(epochs=1,valid_every=1)
    args=SimpleNamespace(**recipe,sharing=recipe['mode'],resume=None)
    out=ROOT/entry['training_path'] if not cli.smoke else ROOT/'reproduction/sota/paper_support/smoke'/cli.id
    out.mkdir(parents=True,exist_ok=True)
'''+source[end:]
source=source.replace('data,entity_maps,roff,n,nr,audit=load_multikg(args.dataset,args.sharing)',
    "data,entity_maps,roff,n,nr,audit,training_arrays,derived=build_inputs(recipe)\n    np.savez_compressed(out/'derived_inputs.npz',**derived)")
source=source.replace("'method':'Aligned reciprocal ComplEx-N3','recipe':vars(args)","'method':'ASRC registered component/perturbation control','recipe':recipe,'id':cli.id,'smoke_only':cli.smoke")
source=source.replace("'input':'published train triples and supplied alignments only; reciprocal train augmentation'",
    "'input':'Registered derived optimization facts and supplied alignments only; full original positive filters retained'")
source=source.replace("Path(__file__).with_name('frozen_data.py')]","Path(__file__).with_name('frozen_data.py'),Path(__file__).with_name('support_data.py')]")
source=source.replace("a=d['arrays']['train'].copy();inverse=a[:,[2,1,0]].copy();inverse[:,1]+=nr",
    "a=training_arrays[kg].copy()\n        if args.reciprocal:\n            inverse=a[:,[2,1,0]].copy();inverse[:,1]+=nr\n        else:\n            inverse=a.copy()  # Match optimizer-step budget by repeating observed orientations.")
assert 'load_multikg(' not in source
(base/'train_support.py').write_text(source,encoding='utf-8')
compile(source,str(base/'train_support.py'),'exec')
print('Generated and syntax-checked train_support.py')
