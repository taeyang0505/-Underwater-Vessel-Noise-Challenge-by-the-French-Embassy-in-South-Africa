"""Install isolated requirements while retaining the pod's CUDA Torch build."""
from pathlib import Path
import hashlib,json,subprocess,sys,time

BASE=Path('/workspace/underwater_vessel_noise_r1')
OPS=BASE/'ops';OPS.mkdir(parents=True,exist_ok=True)
VENV=BASE/'venv';REQ=OPS/'requirements_gpu_cv.txt'
state={'started_unix':time.time(),'status':'installing','base_python':sys.version,
       'requirements_sha256':hashlib.sha256(REQ.read_bytes()).hexdigest()}
def save(): (OPS/'setup_status.json').write_text(json.dumps(state,indent=2))
save()
try:
    if not (VENV/'bin/python').exists():
        subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(VENV)],check=True)
    import torch
    constraint=OPS/'torch_constraint.txt'
    constraint.write_text('torch=='+torch.__version__+'\n')
    py=str(VENV/'bin/python')
    subprocess.run([py,'-m','pip','install','--disable-pip-version-check','--no-cache-dir',
        '-r',str(REQ),'-c',str(constraint)],check=True)
    subprocess.run([py,'-m','pip','check'],check=True)
    subprocess.run([py,'-c','import torch,tabicl,sdm,tabm,numpy,pandas,pyarrow,scipy,sklearn,lightgbm; '
        'print("CUDA_TORCH",torch.__version__,torch.version.cuda,torch.cuda.is_available()); '
        'assert torch.cuda.is_available(); '
        'x=torch.randn(128,128,device="cuda"); print("CUDA_MATMUL",float((x@x.T).mean()))'],check=True)
    state.update(status='passed',finished_unix=time.time(),torch_constraint=constraint.read_text().strip())
except BaseException as exc:
    state.update(status='failed',finished_unix=time.time(),error=type(exc).__name__+': '+str(exc))
    save();raise
save()
