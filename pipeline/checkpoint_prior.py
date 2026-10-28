"""Restore a training-only background prior with its checkpoint binding."""
from pathlib import Path
import hashlib
import json


def attach_checkpoint_prior(model, saved, run, expected_binding=None):
    architecture = saved.get('architecture', {})
    hybrid = architecture.get('architecture_version') in [4,5]
    if not hybrid:
        if saved.get('background_prior') is not None:
            raise ValueError('Background prior supplied for a non-hybrid architecture')
        return model
    if saved.get('kind') != 'cnn' or architecture.get('input_channels') != 8:
        raise ValueError('Background hybrid requires an explicit eight-channel CNN checkpoint')
    path = Path(run) / 'cnn_background_prior.json'
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing regular frozen CNN background prior')
    raw = path.read_bytes(); checksum = hashlib.sha256(raw).hexdigest()
    if checksum != saved.get('background_prior_sha256'):
        raise ValueError('CNN background prior checksum differs from checkpoint')
    prior = json.loads(raw)
    if prior != saved.get('background_prior') or prior.get('version') != 'aggregate_background_prior_v1':
        raise ValueError('CNN background prior payload differs from checkpoint')
    binding = saved.get('background_prior_training_binding', {})
    if prior.get('training_binding') != binding:
        raise ValueError('CNN background prior internal training binding differs from checkpoint')
    for name, value in (expected_binding or {}).items():
        if binding.get(name) != value:
            raise ValueError('CNN background prior belongs to different training data: ' + name)
    if not binding.get('data_package_hash') or not binding.get('train_npz_sha256'):
        raise ValueError('CNN background prior lacks an explicit training-data binding')
    model.background_prior = prior
    return model
