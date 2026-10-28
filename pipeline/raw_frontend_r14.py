"""R14 raw loader + frontend/backbone composition; no original R13 changes."""
import copy
import numpy as np
import torch
from torch import nn
from pipeline.features_r14_torch import TorchFeatureFrontend,VERSION


class RawR14Dataset:
    """Use parent.raw_item without its per-item NumPy feature transform.

    raw q/y are copied unchanged as loss targets. They never enter forward.
    Source ordering, epoch synthesis and site loss weights remain the parent
    dataset's policies; this wrapper changes feature execution placement only.
    """
    def __init__(self,parent):
        if not callable(getattr(parent,'raw_item',None)):raise TypeError('Parent raw_item interface required')
        self.parent=parent
        identity=parent.identity() if callable(parent.identity) else parent.identity
        self.identity={'version':VERSION,'loader':'raw_db_mask_targets','parent':copy.deepcopy(identity),
                       'features_computed_in_Dataset':False}
    def __len__(self):return len(self.parent)
    def set_epoch(self,epoch):return self.parent.set_epoch(epoch)
    def epoch_audit(self):return self.parent.epoch_audit()
    def __getitem__(self,index):
        raw=self.parent.raw_item(index)
        db=np.asarray(raw['db'],np.float32).copy()
        out={k:raw[k] for k in ('share','cell_share','presence','site','sample_id')}
        out['cell_share']=np.asarray(raw['cell_share'],np.float32).copy()
        out.update(db=db,observed_mask=np.isfinite(db))
        if 'loss_weight' in raw:out['loss_weight']=np.float32(raw['loss_weight'])
        return out


class RawFeatureMaskModel(nn.Module):
    def __init__(self,backbone,basis=None,*,include_shape=True):
        super().__init__();self.frontend=TorchFeatureFrontend(basis,include_shape=include_shape,pad_to_eleven=True)
        self.backbone=backbone
    def forward(self,db,observed_mask=None,*,return_weights=False):
        # Keep the numerical frontend out of backbone autocast. No raw q/y,
        # share, source identity or site is an accepted forward argument.
        with torch.autocast(device_type=db.device.type,enabled=False):x,w=self.frontend(db,observed_mask)
        outputs=self.backbone(x,w)
        return (outputs,w) if return_weights else outputs
    def load_state_dict(self,state_dict,strict=True,assign=False):
        result=super().load_state_dict(state_dict,strict=strict,assign=assign)
        self.frontend.verify_frozen_buffers();return result
    def metadata(self):
        return {'version':VERSION,'frontend':self.frontend.metadata(),
                'backbone_architecture':self.backbone.architecture(),
                'supervision_enters_forward':False}
