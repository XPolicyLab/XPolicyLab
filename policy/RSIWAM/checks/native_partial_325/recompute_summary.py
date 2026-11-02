#!/usr/bin/env python3
"""Offline, read-only aggregation of the seven pinned result originals. No simulation."""
from pathlib import Path
from decimal import Decimal
import hashlib,json,math
ROOT=Path(__file__).resolve().parent
def no_duplicates(pairs):
 out={}
 for k,v in pairs:
  if k in out:raise ValueError('Duplicate JSON key: '+k)
  out[k]=v
 return out
def read(path):return json.loads(path.read_text(),object_pairs_hook=no_duplicates)
def aggregate():
 manifest=read(ROOT/'RESULT_MANIFEST.json');all_scores=[];total_success=0;rows=[];ids=set()
 for b in manifest['blocks']:
  assert b['task_seed_id'] not in ids;ids.add(b['task_seed_id'])
  path=ROOT/b['result_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==b['result_sha256']
  data=read(path);details=data['details'];n=b['expected_episodes']
  assert len(details)==n==data['eval_time']
  assert set(details)=={str(i) for i in range(n)}
  scores=[];successes=0;layout_ids=[]
  for ordinal in range(n):
   v=details[str(ordinal)]
   assert type(v['success']) is bool
   assert type(v['score']) in (int,float) and math.isfinite(v['score']) and 0<=v['score']<=1
   assert type(v['layout_id']) is int
   successes+=int(v['success']);scores.append(Decimal(str(v['score'])));layout_ids.append(v['layout_id'])
  score_sum=sum(scores,Decimal(0));sr=Decimal(successes)/n;score=score_sum/n
  assert abs(Decimal(str(data['success_rate']))-sr)<Decimal('1e-12')
  assert abs(Decimal(str(data['score']))-score*100)<Decimal('1e-10')
  rows.append({'task':b['task'],'evaluation_seed':b['evaluation_seed'],'source_group':b['source_group'],'episodes':n,'successes':successes,'success_rate_0_1':float(sr),'success_rate_percent':float(sr*100),'score_sum_0_1':float(score_sum),'mean_score_0_1':float(score),'mean_score_0_100':float(score*100),'raw_aggregate_sr_matches_details':True,'raw_aggregate_score_matches_details_times_100':True,'unique_native_layout_ids':len(set(layout_ids)),'result_sha256':b['result_sha256']})
  total_success+=successes;all_scores+=scores
 assert len(ids)==manifest['expected_block_count']==7
 assert len(all_scores)==manifest['expected_total_episodes']==325
 assert sum(r['episodes'] for r in rows if r['source_group']=='external_first_block')==50
 n=len(all_scores);score_sum=sum(all_scores,Decimal(0));score=score_sum/n;sr=Decimal(total_success)/n
 return {'status':'ACCEPTED_PARTIAL_LOCAL_RESULTS_NOT_FULL_SUITE','selection':'Seven accepted completed task-seed blocks from the frozen 2026-11-02 04:54:56.236969+00:00 snapshot; six queue blocks (275) plus already-counted external first block (50).','candidate_id':manifest['candidate_id'],'recorded_adapter_revision':manifest['recorded_adapter_revision'],'model_seed':1,'evaluation_seeds':[0],'completed_task_seed_blocks':len(rows),'valid_episodes':n,'successes':total_success,'success_rate_0_1':float(sr),'success_rate_percent':float(sr*100),'score_sum_0_1':float(score_sum),'mean_score_0_1':float(score),'mean_score_0_100':float(score*100),'aggregation':'Micro-average over all 325 existing episode records; not an unweighted average of seven block means. No missing scores filled with zero.','score_units':'details[*].score is 0–1; raw top-level score is 100 times its episode mean. Raw top-level success_rate is 0–1.','all_result_sha256_checks_pass':True,'all_raw_aggregate_checks_pass':True,'per_task_seed':rows,'excluded_pending_episode_counts':[6,35],'full_campaign_complete':False,'full_benchmark_score':None,'full_benchmark_sr':None,'recovery_sr':None,'limitations':['Selectively completed task blocks at this progress cut; uneven 50/25 block sizes and seed 0 only. This is not a representative or full benchmark aggregate.','No matched baseline or ordinary-LoRA comparison in this bundle. Task success is not Recovery SR.','Only completed accepted blocks are counted; in-progress records and other seeds/tasks are excluded, not treated as failures.','Existing native result layout_id values are retained. They are not the separate canonical210 experiment layout IDs. Native episode ordering/randomization is not reconstructed here.','Native evidence records actual_initial_physics_snapshot_available=false; no exact full physical initial-state parity claim is made.','Video SHA references are existing completion-time metadata; video bytes were not downloaded or rehashed during this collection.','Recorded code revision and constructor/candidate identity are checked against source records. This collection does not re-audit remote code bytes or establish cross-device bitwise parity.','These local results do not establish official leaderboard acceptance or a literal encoded launcher pass.']}
if __name__=='__main__':print(json.dumps(aggregate(),indent=2,ensure_ascii=False))
