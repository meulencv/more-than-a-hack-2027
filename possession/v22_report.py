"""Consolidate finished external experiments, including failures and adoption rules."""
import hashlib,json
from pathlib import Path
import numpy as np

def mean(rows,kind,metric):
    use=[r[metric] for r in rows if r['kind']==kind]
    return float(np.mean(use)) if use else None

def main():
    sources=[('decoder','data/work/v17/decoder_selection.json'),('specialists','data/work/v17/specialists/results.json'),
        ('highrate','data/work/v17/highrate/results.json'),('conditional_mixture','data/work/v17/gates/results.json'),
        ('typed_transitions','data/work/v18/results.json'),('highrate_long','data/work/v19/results.json'),
        ('explicit_duration','data/work/v20/results.json'),('error_corrector','data/work/v21/results.json'),('sequence_corrector','data/work/v22/results.json')]
    experiments=[];variants=0
    for name,path in sources:
        r=json.loads(Path(path).read_text());chosen=r['chosen'];variants+=len(r['selection_means'])-1
        selection=r['selection_means'];confirmation=r['confirmation_rows']
        result=dict(name=name,source=path,selection_means=selection,chosen=chosen,
            selected_spell_delta=selection[chosen]['spell_f1_mean']-selection['baseline']['spell_f1_mean'],confirmation=None,adopt=False)
        if chosen!='baseline':
            delta={game:float(np.mean([s['spell_f1_mean'] for s in confirmation if s['kind']==chosen and s['game']==game])-np.mean([s['spell_f1_mean'] for s in confirmation if s['kind']=='baseline' and s['game']==game])) for game in sorted(set(s['game'] for s in confirmation))}
            spell_a=mean(confirmation,'baseline','spell_f1_mean');spell_b=mean(confirmation,chosen,'spell_f1_mean')
            macro_a=mean(confirmation,'baseline','macro_f1');macro_b=mean(confirmation,chosen,'macro_f1')
            result['confirmation']=dict(baseline_spell=spell_a,candidate_spell=spell_b,baseline_macro=macro_a,candidate_macro=macro_b,spell_delta_by_game=delta)
            result['adopt']=bool(len(delta)==2 and all(d>0 for d in delta.values()) and macro_b>=macro_a-.002)
        experiments.append(result)
    checkpoints=[]
    patterns=[('data/work/v17/specialists','fold*.joblib',3),('data/work/v17/highrate','fold*.pt',1),('data/work/v18','fold*.pt',1),('data/work/v19','fold*.pt',1),
        ('data/work/v21','fold*/inner_half*.pt',1),('data/work/v21','fold*/correctors.joblib',2),('data/work/v22','fold*.pt',1)]
    for root,pattern,n in patterns:
        for p in sorted(Path(root).glob(pattern)):
            checkpoints.append(dict(path=str(p),models_or_continuations=n,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    audit=json.loads(Path('data/work/v17/error_summary.json').read_text())
    report=dict(experiments=experiments,alternative_configurations=variants,training_fits_including_continuations=sum(c['models_or_continuations'] for c in checkpoints),
        checkpoints=checkpoints,error_diagnosis=audit,challenge_labels_read=False,closed_match_reopened=False,
        canonical_csv_sha256=hashlib.sha256(Path('evaluation/results/final_candidate.csv').read_bytes()).hexdigest(),
        canonical_external_spell_f1=.6757751482502583,canonical_external_macro_f1=.9219784318951262,
        interpretation='All figures here concern external development matches. Selection=J03WOH/J03WPY, second partition=J03WOY/J03WQQ, three observation regimes each; not six independent samples. These matches were used historically and the partitions are reused across hypotheses. Scores are not an unbiased final test or competition score. No new method passes the preregistered adoption rule.' if not any(e['adopt'] for e in experiments) else 'Review adopted method before changing the final artifact.')
    Path('docs/EXPERIMENTOS_V17_V22.json').write_text(json.dumps(report,indent=2))
    print('VARIANTS',variants,'FITS',report['training_fits_including_continuations'],'ADOPTED',[e['name'] for e in experiments if e['adopt']])
    for e in experiments:print(e['name'],e['chosen'],round(e['selected_spell_delta'],6),e['confirmation'])

if __name__=='__main__':main()
