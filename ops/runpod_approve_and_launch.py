"""Start the already-authorized CV run only after the final server gates."""
from pathlib import Path
import json,subprocess,time

BASE=Path('/workspace/underwater_vessel_noise_r1');PROJECT=BASE/'project';OPS=BASE/'ops'
PY=BASE/'venv/bin/python'


def main():
    setup=json.loads((OPS/'setup_status.json').read_text());assert setup['status']=='passed'
    nn=json.loads((PROJECT/'reports/gpu_preflight_r1_server.json').read_text())
    assert nn['CUDA_measured'] and nn['exit_code']==0 and not nn['full_training_executed']
    for folder in ['probe_tabicl16k','probe_kumo4096_fp32_pass']:
        result=json.loads((OPS/folder/'manifest.json').read_text())
        assert result['status']=='completed foundation CV phase' and result['phase']=='inner'
        assert result['per_query_independence_checked'] and not result['outer_accessed']
    if (OPS/'training_state.json').exists():raise FileExistsError('A job already has a state; inspect before relaunch')
    if (PROJECT/'models/group_cv_physics_v2_r1_adam').exists():raise FileExistsError('A model run already exists')
    checks=[('runtime_release_verification.log',[str(PY),'prepare_physics_cv_delivery.py','--verify-delivery',
        '--manifest','delivery/group_cv_physics_v2_r1_gpu_fp32_manifest.json']),
        ('runtime_training_preflight.json',[str(PY),'-m','pipeline.train_group_cv','--config','configs/group_cv_physics_training.json'])]
    for log,command in checks:
        with (OPS/log).open('x') as f:subprocess.run(command,cwd=PROJECT,stdout=f,stderr=subprocess.STDOUT,check=True)
    probe=subprocess.run(['tmux','has-session','-t','underwater_train'],capture_output=True)
    if probe.returncode==0:raise RuntimeError('An existing training session must not be duplicated')
    shell=f'{PY} {OPS / "runpod_train_supervisor.py"} > {OPS / "supervisor.log"} 2>&1'
    subprocess.run(['tmux','new-session','-d','-s','underwater_train',shell],check=True)
    receipt={'status':'launched','unix_time':time.time(),'tmux_session':'underwater_train',
        'runtime_manifest':'delivery/group_cv_physics_v2_r1_gpu_fp32_manifest.json',
        'full_CV_requested':True,'official_submission_requested':False}
    (OPS/'launch_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
