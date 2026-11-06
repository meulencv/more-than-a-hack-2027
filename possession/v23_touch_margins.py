"""Evaluate time margins against unchanged official metrics on external games."""
import hashlib,json
from pathlib import Path
import numpy as np
from possession import v13_cv
from possession.v6_experiment import as_match
from possession.v14_resolution import decode_fine
from possession.benchmark_external import evaluate_prediction
from possession.touch_confirmation import apply_policy,dense_evidence
from evaluation import possession_metrics as pm

ROOT = Path('data/work/v23')
PROTOCOL = Path('docs/PROTOCOLO_V23.json')


def policies():
    out = {'baseline': {'kind': 'baseline'}}
    for seconds in [.2, .4, .8]:
        out[f'hard_merge_{seconds}'] = dict(kind='merge', seconds=seconds, weak_only=False, diagnostic_only=True)
    for seconds in [.4, .8]:
        out[f'weak_merge_{seconds}'] = dict(kind='merge', seconds=seconds, weak_only=True)
    for seconds in [.12, .24, .4, .8]:
        out[f'confirm_{seconds}'] = dict(kind='confirm', seconds=seconds)
    for seconds in [.24, .4]:
        out[f'adaptive_{seconds}'] = dict(kind='confirm', seconds=seconds, adaptive=True)
    for seconds in [.4, .8, 1.2]:
        out[f'recover_{seconds}'] = dict(kind='recover', seconds=seconds)
    return out


def protocol():
    return dict(purpose='Time-margin handling of short interruptions: retrospective confirmation, weak-only merging, and selective recovery of suppressed opposing control pulses.',
        policies=policies(),evidence='Unchanged four-expert OOF probabilities, standard boundary head, automatic physical control scores and actual ball visibility. Never use reference labels in a policy.',
        semantics='No official minimum duration. A true brief controlled H/A spell must remain. No policy changes any D/X frame or merges across dead ball/halves. Strong probability evidence can bypass the confirmation wait.',
        selection_games=v13_cv.settings()['test_folds'][1],check_games=v13_cv.settings()['test_folds'][2],
        selection='Highest mean official spellF1, then macroF1, among non-diagnostic policies, with macro >= baseline-.002. Freeze winner before checking the second partition.',
        adoption='Winner must improve mean spellF1 in EACH of the two check games, preserve check macro >= baseline-.002 and change no D/X frame. Otherwise retain canonical model/CSV.',
        limitation='These external games have been used historically and partitions reused across hypotheses. This is development validation, not an untouched final test or competition score.',
        challenge_labels_read=False,locked_sets_opened=False)


def check_invariants(base, pred, blocks):
    immutable=base>=2
    if not np.array_equal(base[immutable],pred[immutable]): raise AssertionError('Dead ball or unknown frames altered')
    if not np.isin(pred[~immutable],[0,1]).all(): raise AssertionError('Invented dead ball or unknown state')
    inside=np.zeros(len(base),bool)
    for _,a,b in blocks:inside[a:b+1]=True
    if not np.array_equal(base[~inside],pred[~inside]): raise AssertionError('Outside-half frames altered')


def interruptions(segments, seconds):
    out=[]
    for i in range(1,len(segments)-1):
        a,b,k=segments[i];left=segments[i-1];right=segments[i+1]
        if k<2 and left[2]<2 and left[2]==right[2] and k!=left[2] and left[1]==a and b==right[0] and b-a<=round(25*seconds):out.append((a,b,k))
    return out


def pulse_diagnostics(gt_labels,pred):
    gt=np.where(gt_labels>=0,gt_labels,3);valid=pm.valid_runs(gt)
    gs=pm.segments(gt,valid);ps=pm.segments(pred,valid);gg=pm.possession_spells(gt,valid);pp=pm.possession_spells(pred,valid)
    gm={s:i for i,s in enumerate(gg)};pmap={s:i for i,s in enumerate(pp)};mg=set();mp=set();pairs=[]
    for iou,i,j in pm.spell_overlaps(gg,pp):
        if iou<.5:break
        if i in mg or j in mp:continue
        mg.add(i);mp.add(j);pairs.append((i,j))
    out={}
    for tau in [.4,.8,2.]:
        g={gm[z] for z in interruptions(gs,tau)};p={pmap[z] for z in interruptions(ps,tau)}
        out[str(tau)]=dict(gt_pulses=len(g),pred_pulses=len(p),pred_matches_any_gt=len(p&mp),
            strict_pulse_matches=sum(i in g and j in p for i,j in pairs),gt_pulses_matched_by_any_pred=len(g&mg))
    return out


def evaluate(fold,names,with_diagnostics=False):
    rows=[]
    for game in v13_cv.settings()['test_folds'][fold]:
        for mode in ['clean','moderate','stress']:
            r=v13_cv.record(game,mode);s=np.load(v13_cv.ROOT/f'fold{fold}'/f'{game}_{mode}.npz')
            if not np.array_equal(r['frames'],s['frames']):raise ValueError('Timeline mismatch')
            base=decode_fine(s['combined'],s['edges'],s['frames'],as_match(r))
            ev=dense_evidence(s['combined'],s['edges'],s['frames'],r)
            for name in names:
                pred=apply_policy(base,ev,r['blocks'],policies()[name]);check_invariants(base,pred,r['blocks'])
                row=dict(game=game,mode=mode,policy=name,frames_changed=int((pred!=base).sum()),**evaluate_prediction(r['full_labels'],pred))
                if with_diagnostics:row['pulse_diagnostics']=pulse_diagnostics(r['full_labels'],pred)
                rows.append(row);print('RESULT',fold,game,mode,name,round(row['spell_f1_mean'],6),'changed',row['frames_changed'],flush=True)
                if with_diagnostics:np.savez_compressed(ROOT/f'{game}_{mode}_{name}.npz',labels=pred)
    return rows


def means(rows):
    return {name:{key:float(np.mean([r[key] for r in rows if r['policy']==name])) for key in ['spell_f1_mean','macro_f1','frames_changed']} for name in dict.fromkeys(r['policy'] for r in rows)}


def main():
    ROOT.mkdir(parents=True,exist_ok=True);p=protocol()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text())!=p:raise ValueError('Protocol changed')
    PROTOCOL.write_text(json.dumps(p,indent=2))
    selection=evaluate(1,list(policies()));summary=means(selection)
    eligible=[k for k in summary if not policies()[k].get('diagnostic_only') and summary[k]['macro_f1']>=summary['baseline']['macro_f1']-.002]
    chosen=max(eligible,key=lambda k:(summary[k]['spell_f1_mean'],summary[k]['macro_f1']))
    result=dict(selection_rows=selection,selection_means=summary,chosen=chosen,check_rows=[],check_summary={},adopt=False,
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),challenge_labels_read=False,locked_sets_opened=False)
    path=ROOT/'results.json';path.write_text(json.dumps(result,indent=2));print('CHOSEN',chosen,summary[chosen],flush=True)
    if chosen!='baseline':
        check=evaluate(2,['baseline',chosen],with_diagnostics=True);result['check_rows']=check;result['check_summary']=means(check)
        delta={game:float(np.mean([r['spell_f1_mean'] for r in check if r['policy']==chosen and r['game']==game])-np.mean([r['spell_f1_mean'] for r in check if r['policy']=='baseline' and r['game']==game])) for game in p['check_games']}
        result['check_game_deltas']=delta;ms=result['check_summary']
        result['adopt']=bool(all(d>0 for d in delta.values()) and ms[chosen]['macro_f1']>=ms['baseline']['macro_f1']-.002)
    diagnostic=evaluate(0,['baseline'] if chosen=='baseline' else ['baseline',chosen],with_diagnostics=True)
    result['diagnostic_rows']=diagnostic;path.write_text(json.dumps(result,indent=2));print('ADOPT',result['adopt'],result.get('check_game_deltas'),flush=True)


if __name__=='__main__':main()
