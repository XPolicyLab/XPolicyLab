"""Joint RGB assignment under the public one-of-each loose-digit inventory."""
import numpy as np
import copy
from scipy.optimize import linear_sum_assignment

def _assign(pieces):
 loose=[p for p in pieces if p['kind']=='number' and p['slot'] is None]
 if len(loose)!=10:return dict(available=False,reason='observed_loose_digit_count',count=len(loose))
 costs=np.full((10,10),1e6);edges={}
 for i,p in enumerate(loose):
  for c in p['supported']:
   j=int(c['label']);costs[i,j]=c['cost'];edges[i,j]=dict(c)
 rows,cols=linear_sum_assignment(costs)
 if any((int(i),int(j)) not in edges for i,j in zip(rows,cols)):
  return dict(available=False,reason='no_supported_perfect_assignment')
 best=float(costs[rows,cols].sum());runner=None
 for i,j in zip(rows,cols):
  probe=costs.copy();probe[i,j]=1e6;a,b=linear_sum_assignment(probe)
  if np.max(probe[a,b])>=1e6:continue
  v=float(probe[a,b].sum());runner=min(runner,v) if runner is not None else v
 if runner is not None and runner-best<=1e-6:return dict(available=False,reason='nonunique_visual_assignment',best_cost=best,runner_up_cost=runner)
 assignments=[]
 for i,j in zip(rows,cols):
  p=loose[i];p['independent_supported']=p['supported'];p['supported']=[edges[i,j]]
  assignments.append(dict(piece=p['id'],label=str(j),visual_cost=costs[i,j]))
 return dict(available=True,method='public_digit_inventory_minimum_RGB_cost',best_cost=best,runner_up_cost=runner,total_cost_margin=runner-best if runner is not None else None,assignments=assignments)


def _supported_assignment(pieces):
 report=_assign(pieces)
 if report.get('reason')!='no_supported_perfect_assignment':return report
 # Retry only an infeasible visual inventory. The rank cap must not erase a
 # fourth candidate that already meets the unchanged RGB confidence margin.
 # Neither the equation nor its expected missing symbol enters this fallback.
 trial=copy.deepcopy(pieces);added=[]
 for p in trial:
  ranked=p.get('candidates',[])
  if p['kind']!='number' or p['slot'] is not None or not p['supported'] or len(ranked)<4:continue
  candidate=dict(ranked[3]);labels={c['label'] for c in p['supported']}
  if candidate['label'] in labels:continue
  if candidate['cost']>min(.65,ranked[0]['cost']+.08):continue
  if (p.get('interior') or {}).get('paired') and candidate['label'] in ('6','9'):
   candidate['cost']+=.03
  if candidate['cost']>min(c['cost'] for c in p['supported'])+.08:continue
  p['supported'].append(candidate);added.append(dict(piece=p['id'],candidate=candidate))
 if not added:return report
 repaired=_assign(trial)
 if not repaired.get('available'):return report
 margin=repaired.get('total_cost_margin')
 if margin is not None and margin<.05:return dict(report,fallback_rejected='weak_global_visual_margin',fallback_margin=margin)
 for original,new in zip(pieces,trial):
  original['supported']=new['supported']
  if 'independent_supported' in new:original['independent_supported']=new['independent_supported']
 repaired.update(fallback='infeasible_top3_inventory_only',added_visual_candidates=added,initial_assignment=report)
 return repaired


def assign(pieces):
 # First preserve the original RGB inventory assignment. Reconcile only if
 # its chosen 5 contradicts a strong, independent lower-counter cue for 6.
 # No equation, answer, layout ID or score enters this decision.
 original={p['id']:[dict(c) for c in p['supported']] for p in pieces}
 report=_supported_assignment(pieces)
 conflicts=[]
 for p in pieces:
  cue=p.get('counter_audit') or {};allowed={c['label'] for c in (p.get('counter_support') or [])}
  if p['kind']=='number' and p['slot'] is None and len(p['supported'])==1 and p['supported'][0]['label']=='5' and cue.get('applied') and cue.get('cue')=='enclosed_lower_counter' and allowed=={'6'}:
   conflicts.append(p['id'])
 if not conflicts:return report
 for p in pieces:
  p['supported']=[dict(c) for c in original[p['id']]]
  p.pop('independent_supported',None)
  ranked=p['candidates'];labels={c['label'] for c in p['supported']}
  if p['kind']=='number' and p['slot'] is None and len(ranked)>3 and ranked[3]['label']=='5' and labels<={'6','8','9'} and ranked[3]['cost']<=min(.65,ranked[0]['cost']+.08):
   p['supported'].append(dict(ranked[3]))
  if p['id'] in conflicts:p['supported']=[dict(c) for c in p['counter_support']]
 corrected=_assign(pieces)
 corrected['counter_conflicts']=conflicts
 corrected['initial_assignment']=report
 corrected['reconciliation']='lower_counter_conflicts_with_assigned_5; admit RGB-margin rank4 5 and recompute'
 return corrected
