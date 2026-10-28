"""Stdlib synthetic-files fixture; no model/query/training execution."""
from pathlib import Path
from tempfile import TemporaryDirectory
import csv,hashlib,html,json,sys
from ops import render_r14_production_status_v2 as ui

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False)+'\n')
def generated(root,name):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['ID','share_added_vessel','score_added_vessel'])
        w.writerows((f'NUMERIC_FIXTURE_{i:04d}',.2,.7) for i in range(1500))
    write(p.with_suffix('.manifest.json'),{'path':name,'rows':1500,'ID_order_preserved':True,
        'numeric_roundtrip_exact':True,'official_upload':False,'output_sha256':sha(p),'checks':{'fixture_check':True}})
    return p

def run():
    checks=[];original=ROOT/'ops/render_r14_production_status.py';original_sha=sha(original)
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            name=Path(args[0]).name.lower()
            if name in ('test.csv','sample_submission.csv','share.txt','presence.txt') or Path(name).suffix in ('.pt','.npz','.wav','.flac'):
                raise AssertionError('Forbidden model/query/audio access:'+name)
        if event in ('socket.connect','subprocess.Popen','os.system'):raise AssertionError('No process/network actions')
    sys.addaudithook(guard)
    with TemporaryDirectory(prefix='r14_status_r2_numeric_') as temp:
        r=Path(temp);nn=r/'models/r14_production_nn_fold0_r1';old=r/'reports/r14_production';new=r/'reports/r14_production_r2'
        write(r/'configs/r14_production_mac_v1.json',{'production_recipe_approved':True,'models':{k:{'epochs':50 if k==ui.KINDS[-1] else 100,'device':'cpu' if k=='itransformer' else 'mps'} for k in ui.KINDS}})
        for kind in ui.KINDS:
            rows=[{'epoch':6 if kind==ui.KINDS[-1] else 100,'best_selection_rmse':.1,
                'train_online_rmse':.2,'selection':{'rmse':.1},'actual_epoch_seconds_including_validation_checkpoint':10}]
            p=nn/kind/'epochs.jsonl';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('\n'.join(json.dumps(v) for v in rows)+'\n')
            if kind!=ui.KINDS[-1]:write(nn/kind/'summary.json',{'status':'completed','epochs_run':100,'best_selection_rmse':.1})
        write(r/'models/r14_production_trees_r1/ALL_FIVEFOLD_TREES_FROZEN.json',{'status':'completed','models':{str(i):{} for i in range(5)}})
        old_error='R1 output stopped <script>alert(1)</script>'
        write(r/'ops/r14_production_state.json',{'status':'failed','updated_KST':'FIXTURE_R1_TIME','error':old_error})
        state={'status':'running','production_recipe_approved':True,'updated_KST':'FIXTURE_R2_TIME',
            'neural_current_batch':{'kind':ui.KINDS[-1],'epoch':7,'rows_seen':128,'total_rows':1000},
            'neural_run_state':{'phase':'TF'},'stages':{}}
        write(r/'ops/r14_production_state_r2.json',state)
        write(old/'TREE_FIVEFOLD_REPORT.json',{'status':'completed','pooled_OOF':{'rmse':.1,'auc_score':.9},'folds':[]})
        generated(r,'submissions/R14/submission_r14_fulltree_5fold.csv')
        generated(r,'submissions/R14/submission_r14_fold0.csv')  # must NOT be used by R2 hybrid links
        generated(r,'submissions/R14_r2/submission_r14_fold0.csv')
        generated(r,'submissions/r14_main_r2/submission_r14_step1.csv')
        bad=generated(r,'submissions/R14_r2/submission_r14_fold0_with_tfgrid.csv')
        bad.with_suffix('.manifest.json').unlink()
        for d,token in [(old,'OLD_HYBRID_SENTINEL'),(new,'R2_HYBRID_SENTINEL')]:
            d.mkdir(parents=True,exist_ok=True)
            with (d/'HYBRID_fast3_17_PANEL_METRICS.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['panel','rmse','auc_score']);w.writeheader();w.writerow({'panel':token,'rmse':.1,'auc_score':.9})
            write(d/'HYBRID_fast3_QUERY_DISTRIBUTION_AUDIT.json',{'R9':{},'new':{}})
        out=ui.render(root=r);text=out.read_text()
        check('same_canonical_output_default',out==r.resolve()/'reports/R14_PRODUCTION_START.html')
        check('R2_state_selected', 'FIXTURE_R2_TIME' in text)
        check('old_supervisor_failure_is_archived_not_training_error','<h1>본체 학습 진행 중</h1>' in text and '이전 R1 출력 감독 종료' in text)
        check('old_error_escaped',old_error not in text and html.escape(old_error,quote=True) in text)
        check('R1_tree_report_preserved','r14_production/TREE_FIVEFOLD_REPORT.json' in text)
        check('R1_tree_download_preserved','../submissions/R14/submission_r14_fulltree_5fold.csv' in text)
        check('R2_hybrid_reports_only','R2_HYBRID_SENTINEL' in text and 'OLD_HYBRID_SENTINEL' not in text)
        check('R2_exact_Fold0_download','../submissions/R14_r2/submission_r14_fold0.csv' in text)
        check('R2_bagging_download','../submissions/r14_main_r2/submission_r14_step1.csv' in text)
        check('old_hybrid_download_not_reused','../submissions/R14/submission_r14_fold0.csv' not in text)
        check('unverified_TF_output_hidden','href="../submissions/R14_r2/submission_r14_fold0_with_tfgrid.csv"' not in text)
        check('R1_tree_stage_completed_without_R2_stage_entry','Tree5 단독 CSV · R1 보존</b><span>완료' in text)
        check('neural_and_tree_counts_derived','10 / 10' in text and '3 / 3' in text and '6 / 50' in text)
        before=json.dumps(state,sort_keys=True);ui.render(state,root=r)
        check('caller_snapshot_not_mutated',json.dumps(state,sort_keys=True)==before)
        p=r/'submissions/R14_r2/submission_r14_fold0.csv';p.write_text(p.read_text()+'corrupt\n')
        check('changed_generated_CSV_rejected',ui.validate_submission(p,r) is None)
        text=ui.render(state,root=r).read_text()
        check('changed_CSV_download_hidden','href="../submissions/R14_r2/submission_r14_fold0.csv"' not in text)
        (r/'ops/r14_production_state_r2.json').unlink();text=ui.render(root=r).read_text()
        check('missing_R2_state_falls_back_to_training_files','실제 학습 파일 fallback' in text and '최신 R2 감독 상태 생성 대기' in text)
    check('old_renderer_source_unchanged',sha(original)==original_sha)
    result={'format':'r14_status_renderer_R2_synthetic_fixture_v1','status':'passed','checks':checks,'checks_passed':len(checks),
        'code_sha256':{'ops/render_r14_production_status.py':original_sha,'ops/render_r14_production_status_v2.py':sha(ROOT/'ops/render_r14_production_status_v2.py'),
                       'ops/verify_r14_status_v2.py':sha(__file__)},
        'synthetic_files_only':True,'model_training_inference_raw_Test_or_audio_accessed':False,
        'R1_models_config_state_reports_changed':False,'default_page_only_actual_write_when_invoked':True}
    path=ROOT/'reports/R14_STATUS_RENDERER_V2_FIXTURE.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','checks':len(checks),'output':str(path),'sha256':sha(path)}))

if __name__=='__main__':run()
