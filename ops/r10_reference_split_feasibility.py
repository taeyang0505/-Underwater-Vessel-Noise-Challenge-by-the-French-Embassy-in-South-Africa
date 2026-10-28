"""Metadata-only hull/day reference split feasibility; never changes R9 pool.

No acoustic values, weak classes, q/y/presence labels, model results, Test,
report arrays, training or external access are read. Exact dynamic programming
optimizes retained clip count CONDITIONAL ON EACH fixed day partition; it
does not claim a globally optimal balanced graph separator.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
COLUMNS=['reference_row_index','clip_id','hull_id','recording_day','utc_begin','utc_end',
         'duration_seconds','native_complete_minute_count','sha256']
HASH_SEEDS=(2026102601,2026102602,2026102603,2026102611,2026102612,2026102613,
            2026102621,2026102622,2026102623,2026102631,2026102632,2026102633)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def token(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def utc():return datetime.now(timezone.utc).isoformat()


def graph_audit(frame):
    import networkx as nx
    import numpy as np
    g=nx.Graph();pairs={}
    for row in frame.itertuples():
        key=('h:'+row.hull_id,'d:'+row.recording_day)
        g.add_edge(*key);pairs.setdefault(key,[]).append(row.clip_id)
    clip=nx.Graph();clip.add_nodes_from(frame.clip_id)
    criteria={'same_hull':0,'same_recording_day':0,'same_original_WAV_bytes':0,'UTC_interval_overlap':0}
    rows=list(frame.itertuples())
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            flags=[a.hull_id==b.hull_id,a.recording_day==b.recording_day,a.sha256==b.sha256,
                   a.begin_ns<b.end_ns and b.begin_ns<a.end_ns]
            for name,present in zip(criteria,flags):criteria[name]+=int(present)
            if any(flags):clip.add_edge(a.clip_id,b.clip_id)
    bridges=[]
    for u,v in nx.bridges(g):
        key=(u,v) if u.startswith('h:') else (v,u)
        reduced=g.copy();reduced.remove_edge(u,v)
        components=list(nx.connected_components(reduced));lookup={node:i for i,nodes in enumerate(components) for node in nodes}
        remaining=frame[~frame.clip_id.isin(pairs[key])]
        counts={}
        for row in remaining.itertuples():
            c=lookup['h:'+row.hull_id];counts[c]=counts.get(c,0)+1
        sizes=sorted(counts.values(),reverse=True)
        bridges.append({'hull':key[0][2:],'day':key[1][2:],'clips_at_pair':len(pairs[key]),
                        'single_clip_multigraph_bridge':len(pairs[key])==1,
                        'hull_degree':g.degree(key[0]),'day_degree':g.degree(key[1]),
                        'remaining_nonempty_clip_component_sizes':sizes,
                        'meaningful_two_source_split':len(sizes)>=2})
    articulations=list(nx.articulation_points(g));clip_articulations=[]
    for clip_id in nx.articulation_points(clip):
        reduced=clip.copy();reduced.remove_node(clip_id)
        clip_articulations.append({'clip_id':clip_id,'remaining_component_sizes':sorted([len(c) for c in nx.connected_components(reduced)],reverse=True)})
    facts={'clips':len(frame),'unique_hulls':frame.hull_id.nunique(),'recording_days':frame.recording_day.nunique(),
           'distinct_hull_day_pairs':g.number_of_edges(),'bipartite_nodes':g.number_of_nodes(),
           'bipartite_components':nx.number_connected_components(g),
           'clip_union_graph_components':nx.number_connected_components(clip),
           'clip_union_graph_edges':clip.number_of_edges(),'clip_graph_edge_criteria_counts':criteria,
           'hull_articulation_nodes':[x[2:] for x in articulations if x.startswith('h:')],
           'day_articulation_nodes':[x[2:] for x in articulations if x.startswith('d:')],
           'simple_graph_bridge_pair_count':len(bridges),'bridge_pair_details':bridges,
           'bridge_pairs_hull_degree1':sum(v['hull_degree']==1 for v in bridges),
           'bridge_pairs_whose_deletion_leaves_two_nonempty_clip_components':sum(v['meaningful_two_source_split'] for v in bridges),
           'single_clip_articulation_count':len(clip_articulations),'single_clip_articulation_details':clip_articulations,
           'duplicate_original_WAV_bytes':int(frame.sha256.duplicated().sum())}
    return facts,clip


def partitions(frame):
    import numpy as np
    days=sorted(frame.recording_day.unique());counts=frame.groupby('recording_day').size().to_dict();n=len(frame)
    result=[]
    def add(name,a):
        a=set(a);b=set(days)-a
        if a and b:result.append((name,a,b))
    closest=min(range(1,len(days)),key=lambda k:(abs(sum(counts[d] for d in days[:k])-n/2),k))
    add('chronological_closest_raw_clip_half',days[:closest])
    add('chronological_alternating_days',days[::2])
    firstyear=min(d[:4] for d in days);add('earliest_year_vs_later_years',[d for d in days if d[:4]==firstyear])
    for seed in HASH_SEEDS:
        order=sorted(days,key=lambda d:(token([seed,d]),d));sides=[[],[]];loads=[0,0]
        for day in order:
            side=min([0,1],key=lambda i:(loads[i],len(sides[i]),token([seed,day,i])))
            sides[side].append(day);loads[side]+=counts[day]
        add(f'hash_greedy_raw_clip_balance_seed{seed}',sides[0])
    # One predeclared metadata graph heuristic, not acoustic/model score search.
    matrix=np.zeros((len(days),len(days)));lookup={d:i for i,d in enumerate(days)}
    for _,group in frame.groupby('hull_id',sort=True):
        c=group.groupby('recording_day').size().to_dict()
        for a in c:
            for b in c:
                if a!=b:matrix[lookup[a],lookup[b]]+=min(c[a],c[b])
    eig,vec=np.linalg.eigh(np.diag(matrix.sum(1))-matrix);fiedler=vec[:,1]
    if next((v for v in fiedler if abs(v)>1e-10),1)<0:fiedler=-fiedler
    order=sorted(days,key=lambda d:(float(fiedler[lookup[d]]),d))
    k=min(range(1,len(days)),key=lambda k:(abs(sum(counts[d] for d in order[:k])-n/2),k))
    add('spectral_day_affinity_closest_raw_clip_half',order[:k])
    return result


def exact_hull_assignment(frame,days_a,days_b):
    """Exact reachable(Aclips,Bclips) DP, allowing hull A/B/drop choices."""
    import numpy as np
    hulls=sorted(frame.hull_id.unique());capacity_a=int(frame.recording_day.isin(days_a).sum());capacity_b=len(frame)-capacity_a
    counts=[]
    for hull in hulls:
        subset=frame[frame.hull_id==hull]
        counts.append((int(subset.recording_day.isin(days_a).sum()),int(subset.recording_day.isin(days_b).sum())))
    state=np.zeros((capacity_a+1,capacity_b+1),bool);state[0,0]=True;history=[state]
    for a,b in counts:
        new=state.copy()
        if a:new[a:,:]|=state[:-a,:]
        if b:new[:,b:]|=state[:,:-b]
        history.append(new);state=new
    ai,bi=np.indices(state.shape);total=ai+bi
    valid=state&(ai>0)&(bi>0)
    balanced=valid&(20*ai>=9*total)&(20*bi>=9*total) # exact45%-55%
    exact=valid&(abs(ai-bi)<=1)
    def choose(mask):
        if not mask.any():return None
        maxn=int(total[mask].max());candidates=np.argwhere(mask&(total==maxn))
        a,b=min(candidates.tolist(),key=lambda x:(abs(x[0]-x[1]),x[0]))
        roles={};x,y=a,b
        for i in range(len(hulls)-1,-1,-1):
            ca,cb=counts[i];previous=history[i]
            if ca and x>=ca and previous[x-ca,y]:roles[hulls[i]]='A';x-=ca
            elif cb and y>=cb and previous[x,y-cb]:roles[hulls[i]]='B';y-=cb
            elif previous[x,y]:roles[hulls[i]]='drop'
            else:raise AssertionError('DP reconstruction failed')
        if x or y:raise AssertionError('DP origin mismatch')
        return {'retained':maxn,'A_clips':a,'B_clips':b,'hull_roles':roles}
    return choose(balanced),choose(exact),sum(max(a,b) for a,b in counts),sum(a>0 and b>0 for a,b in counts)


def summarize(group):
    return {'clips':len(group),'hulls':group.hull_id.nunique(),'days':group.recording_day.nunique(),
            'header_hours':float(group.duration_seconds.sum()/3600),
            'native_complete_minutes':int(group.native_complete_minute_count.sum()),
            'native_complete_minute_hours':float(group.native_complete_minute_count.sum()/60),
            'duration_seconds_min_median_max':[float(group.duration_seconds.min()),float(group.duration_seconds.median()),float(group.duration_seconds.max())] if len(group) else None,
            'year_counts':{str(k):int(v) for k,v in group.recording_day.str[:4].value_counts().sort_index().items()},
            'day_clip_counts':{str(k):int(v) for k,v in group.recording_day.value_counts().sort_index().items()}}


def run(output):
    import numpy as np
    import pandas as pd
    import networkx as nx
    source=ROOT/'data/versions/vtuad_full_r9/reference_manifest.parquet';source_hash=sha(source)
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve existing feasibility results')
    output.mkdir(parents=True);started=time.perf_counter()
    plan={'format':'r10_reference_split_feasibility_plan_v1','created_utc':utc(),
          'source':str(source.relative_to(ROOT)),'source_sha256':source_hash,'columns_read':COLUMNS,
          'partition_families':['chronologicalclosesthalf','alternatingdays','earliestyear','12fixedhashgreedy','spectraldayaffinity'],
          'hash_seeds':list(HASH_SEEDS),'primary_balance_constraint':'eachretainedside45%-55%',
          'secondary_balance_constraint':'abs(retainedA-retainedB)<=1',
          'optimization':'exactclipretentionDPforeachFIXEDdaypartition;notglobalminbalancedseparator',
          'temporal_crossday_policy':'conservativelyexcludeUTCmidnightstraddlingclipsbeforeDP',
          'labels_model_results_acoustic_Test_report_read':False,'production_pool_modified':False}
    write(output/'predeclared_plan.json',plan)
    frame=pd.read_parquet(source,columns=COLUMNS).copy().sort_values('clip_id').reset_index(drop=True)
    if len(frame)!=509 or frame[COLUMNS].isna().any().any() or not frame.clip_id.is_unique:raise ValueError('Expected509completeunique metadata rows')
    begin=pd.to_datetime(frame.utc_begin,utc=True);end=pd.to_datetime(frame.utc_end,utc=True)
    if not (end>begin).all():raise ValueError('Invalid UTC support')
    frame['begin_ns']=begin.astype('int64');frame['end_ns']=end.astype('int64')
    frame['crosses_midnight']=begin.dt.floor('D')!=(end-pd.Timedelta(nanoseconds=1)).dt.floor('D')
    if not np.array_equal(begin.dt.strftime('%Y-%m-%d'),frame.recording_day):raise ValueError('Recording day does not match UTCbegin metadata')
    graph,clip_graph=graph_audit(frame)
    if graph['bipartite_components']!=1 or graph['clip_union_graph_components']!=1:raise AssertionError('Original full graph not one component')
    temporal_excluded=frame[frame.crosses_midnight];pool=frame[~frame.crosses_midnight].copy()
    trials=[]
    for name,days_a,days_b in partitions(pool):
        best,equal,upper,cross_hulls=exact_hull_assignment(pool,days_a,days_b)
        assigned=frame.copy();assigned['day_side']=np.where(assigned.recording_day.isin(days_a),'A','B')
        assigned['role']='excluded';assigned['exclusion_reason']='conservativeUTCmidnightcrossing'
        if best is None:raise ValueError('No nonempty balanced feasible partition')
        roles=best.pop('hull_roles')
        for i,row in assigned.iterrows():
            if row.crosses_midnight:continue
            role=roles[row.hull_id]
            if role==row.day_side:assigned.loc[i,'role']=role;assigned.loc[i,'exclusion_reason']=''
            else:assigned.loc[i,'exclusion_reason']='hull_assigned_to_other_day_partition' if role!='drop' else 'whole_hull_dropped_for_balance'
        a=assigned[assigned.role=='A'];b=assigned[assigned.role=='B'];drop=assigned[assigned.role=='excluded']
        cross_edges=[(u,v) for u,v in clip_graph.edges if (u in set(a.clip_id) and v in set(b.clip_id)) or (u in set(b.clip_id) and v in set(a.clip_id))]
        if set(a.hull_id)&set(b.hull_id) or set(a.recording_day)&set(b.recording_day) or set(a.sha256)&set(b.sha256) or cross_edges:raise AssertionError('Residual hull/day/bytes/UTC overlap')
        if len(a)+len(b)!=best['retained'] or len(a)!=best['A_clips'] or len(b)!=best['B_clips']:raise AssertionError('Reconstructed counts differ')
        assigned.drop(columns=['begin_ns','end_ns']).to_csv(output/(name+'.csv'),index=False)
        if equal:equal.pop('hull_roles')
        trials.append({'name':name,'proposed_day_sets':{'A':sorted(days_a),'B':sorted(days_b)},
            'raw_day_partition_clips':{'A':int(pool.recording_day.isin(days_a).sum()),'B':int(pool.recording_day.isin(days_b).sum())},
            'cross_partition_hulls_before_exclusion':cross_hulls,
            'fixed_partition_max_retained_without_balance':upper,
            'fixed_partition_min_exclusions_without_balance':len(frame)-upper,
            'optimal_retained_45_55':best,'optimal_retained_exact_balance_pm1':equal,
            'excluded_clips':len(drop),'retained_fraction':(len(a)+len(b))/509,
            'A_summary':summarize(a),'B_summary':summarize(b),'excluded_summary':summarize(drop),
            'cross_hull_day_bytes_interval_overlap_violations':0,
            'both_sides_at_least250':len(a)>=250 and len(b)>=250,
            'retained_graph_components':nx.number_connected_components(clip_graph.subgraph([*a.clip_id,*b.clip_id])),
            'metadata_assignment_file':name+'.csv','not_applied_to_production':True})
        print(json.dumps({'partition':name,'A':len(a),'B':len(b),'excluded':len(drop),'elapsed_seconds':time.perf_counter()-started}),flush=True)
    best=max(trials,key=lambda q:(q['optimal_retained_45_55']['retained'],-abs(q['A_summary']['clips']-q['B_summary']['clips']),q['name']))
    eq=max(trials,key=lambda q:q['optimal_retained_exact_balance_pm1']['retained'])
    if sha(source)!=source_hash:raise AssertionError('Original reference manifest changed')
    report={'format':'r10_reference_split_feasibility_results_v1','status':'completed','created_utc':utc(),
            'predeclared_plan_sha256':sha(output/'predeclared_plan.json'),'source_sha256':source_hash,
            'graph':graph,'conservative_midnight_exclusions':temporal_excluded.clip_id.tolist(),
            'tested_partitions':len(trials),'trials':trials,
            'best_tested_45_55':{'name':best['name'],'retained':best['optimal_retained_45_55']['retained'],
                'excluded':best['excluded_clips'],'A':best['A_summary'],'B':best['B_summary']},
            'best_tested_exact_balance_pm1':{'name':eq['name'],**eq['optimal_retained_exact_balance_pm1']},
            'any_tested_250_250':any(t['both_sides_at_least250'] for t in trials),
            'source_model_data_pool_unchanged':True,'model_fitting_or_predictions_executed':False,
            'labels_acoustic_Test_report_model_performance_read':False,
            'limitations':['Exactmaximumisconditionaloneachfixeddaypartition,notaglobalbalancedseparatorproof.',
                'Hull/day/UTC/byteisolationisconditionalonretainedclips;allrecordingsstillshareprovider/receivercorpus.',
                'Articulationdayremovaldeletesmanyclips;simplebipartitebridgepairswithparallelclipsarenotsinglebridgeclips.',
                'Arbitraryclip250/250splitdoesnotguaranteeunseenhullsordays.',
                'MetadatafeasibilitydoesnotdemonstrateimprovedRMSE/AUCandhasnotalteredR9.'],
            'elapsed_seconds':time.perf_counter()-started,'script_sha256':sha(Path(__file__))}
    write(output/'RESULTS.json',report)
    pd.DataFrame([{'partition':t['name'],'A':t['A_summary']['clips'],'B':t['B_summary']['clips'],
                  'retained':t['optimal_retained_45_55']['retained'],'excluded':t['excluded_clips'],
                  'unbalanced_upper':t['fixed_partition_max_retained_without_balance'],
                  'hullsA':t['A_summary']['hulls'],'hullsB':t['B_summary']['hulls'],
                  'daysA':t['A_summary']['days'],'daysB':t['B_summary']['days']} for t in trials]).to_csv(output/'all_partitions.csv',index=False)
    lines=['# R10 참조 hull/day 독립 분할: metadata 실현 가능성','',
        f"원509clips/178hulls/33days는 실제 hull/day와 UTC overlap/bytes union graph에서 **1component**입니다. metadata만 사용했고 weak class, acoustic array, q/y/presence, 모델 성능, Test, 미사용 report는 읽지 않았습니다.",'',
        f"Bipartite articulation은 hull {len(graph['hull_articulation_nodes'])}개/day {len(graph['day_articulation_nodes'])}개입니다. simple graph의bridge pair는{graph['simple_graph_bridge_pair_count']}개이며그중hull degree1은{graph['bridge_pairs_hull_degree1']}개입니다. pair전체clip을제거한뒤실제두비어있지않은clipcomponent를만드는pair는{graph['bridge_pairs_whose_deletion_leaves_two_nonempty_clip_components']}개입니다. 이것을'소수bridgeclips로250/250'이라바꿔말할수없습니다.",'',
        f"Clip union graph의단일clip articulation은{graph['single_clip_articulation_count']}개입니다. UTC자정straddling제외는{len(temporal_excluded)}개였습니다.",'',
        f"사전고정{len(trials)}개daypartition마다hull을A/B/drop에배정하는정확한DP로retention을계산했습니다. 양쪽45%-55%조건의최대tested결과는 **{best['A_summary']['clips']} / {best['B_summary']['clips']}**,총{best['optimal_retained_45_55']['retained']}개유지·**{best['excluded_clips']}개제외**입니다.",'',
        f"정확균형±1개기준의besttested는{eq['optimal_retained_exact_balance_pm1']['A_clips']} / {eq['optimal_retained_exact_balance_pm1']['B_clips']}입니다. tested250/250가능: {report['any_tested_250_250']}.",'',
        '|사전 분할|A|B|유지|제외|','|---|---:|---:|---:|---:|']
    for t in trials:lines.append(f"|{t['name']}|{t['A_summary']['clips']}|{t['B_summary']['clips']}|{t['optimal_retained_45_55']['retained']}|{t['excluded_clips']}|")
    lines+=['','## 해석 한계','',
        '이는각fixeddaypartition안의최적retention이며모든가능한daypartition의전역최적balancedseparator는아닙니다. 따라서tested최선보다좋은분할이수학적으로불가능하다고주장하지않습니다.',
        'Hull/day/UTC/bytes가양쪽에서겹치지않는실제선택부분집합을만들수는있지만,같은receiver/providercorpus를쓰므로완전히새로운해역/센서독립검증이되는것은아닙니다.',
        '현재R9production참조509풀·모델·가중치·학습은전혀변경하지않았습니다. CSV는후보metadata분할이며자동적용하지않습니다. 현재나미래의RMSE/AUC개선을측정한결과도아닙니다.',
        '', '[전체 graph·bridge·각side분포·SHA](RESULTS.json) / [전후보 요약](all_partitions.csv)']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'completed','best':report['best_tested_45_55'],'elapsed_seconds':report['elapsed_seconds']}));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'reports/r10_diagnostics/reference_split_feasibility_r1')
    a=p.parse_args();run(a.output)


if __name__=='__main__':main()
