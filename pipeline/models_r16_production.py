"""Fresh R16 relative8 full mask models and independent temporal classifiers.

The mask backbones retain R9 task adaptations. No old weights, PCA, absolute
dB, LF anchor, q0 or contrast-derived presence loss are used. Classifier
parameters and gradients are entirely independent of the cell mask path.
"""
from copy import deepcopy
import torch
from torch import nn
from pipeline.models_r9 import R9RelativeMaskAdapter, MODEL_DEFAULTS
from pipeline.relative_grid_r6_v2 import FEATURE_NAMES

INPUT_POLICY='r16_relative8_quantized_weights_noPCA_noLFanchor_v1'
SCORE_POLICY='independent_temporal_classifier_sigmoid_v1'
TEMPORAL_CHANNELS=(0,1,2,3,6,5)


class IndependentTemporalClassifier(nn.Module):
    """Same separate 3,745-parameter classifier validated in R16 pilot."""
    def __init__(self,dropout=.1):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(6,16,3,padding=1),nn.GELU(),nn.Dropout2d(dropout),
                                nn.Conv2d(16,16,3,padding=1),nn.GELU())
        self.head=nn.Sequential(nn.Linear(32,16),nn.GELU(),nn.Dropout(dropout),nn.Linear(16,1))
    def forward(self,x):
        obs=x[:,5]>0
        h=self.body(x[:,list(TEMPORAL_CHANNELS)])
        om=obs[:,None].to(h.dtype);n=om.sum((2,3)).clamp_min(1)
        avg=(h*om).sum((2,3))/n
        maximum=h.masked_fill(~obs[:,None],-torch.inf).amax((2,3))
        return self.head(torch.cat([avg,maximum],1))[:,0]


class R16ProductionModel(nn.Module):
    def __init__(self,kind,**settings):
        super().__init__()
        self.regressor=R9RelativeMaskAdapter(kind,**settings)
        self.classifier=IndependentTemporalClassifier(self.regressor.model_config['dropout'])
        self.kind=kind;self.input_policy=INPUT_POLICY;self.score_policy=SCORE_POLICY
    def forward(self,x,weights):
        self.regressor._validate(x,weights)
        q=self.regressor._cell_mask(x).float()
        q=torch.where(weights>0,q,torch.zeros_like(q))
        share=(q*weights.float()).sum((1,2))
        return share,self.classifier(x),q
    def architecture(self):
        return {'format':'r16_relative8_decoupled_mask_v1','architecture_version':16,
                'kind':self.kind,'settings':deepcopy(self.regressor.model_config),
                'input_channels':8,'feature_names':list(FEATURE_NAMES),
                'input_policy':INPUT_POLICY,'score_policy':SCORE_POLICY,
                'classifier_input_channels':list(TEMPORAL_CHANNELS),
                'regression_parameters':sum(p.numel()for p in self.regressor.parameters()),
                'classification_parameters':sum(p.numel()for p in self.classifier.parameters()),
                'total_parameters':sum(p.numel()for p in self.parameters()),
                'shared_regression_classifier_parameters':0,'pretrained':False,
                'requires_background_prior':False}


def make_r16_model(kind,**settings):return R16ProductionModel(kind,**settings)


def decoupled_loss(prediction,targets,weights):
    """Raw q/y stay unchanged; rounding discrepancy is measured, not relabeled."""
    share,logit,q=prediction;y,z,true_q=targets
    if q.shape!=true_q.shape or q.shape!=weights.shape:raise ValueError('Cell targets/weights shape differs')
    regression=(share-y).square()+.25*((q-true_q).square()*weights).sum((1,2))
    classification=nn.functional.binary_cross_entropy_with_logits(logit,z,reduction='none')
    if not torch.isfinite(regression).all()or not torch.isfinite(classification).all():raise FloatingPointError('Nonfinite R16 loss')
    return regression.mean()+classification.mean(),{'share_mse':(share-y).square(),
        'weighted_cell_mse':((q-true_q).square()*weights).sum((1,2)),
        'classification_bce':classification,
        'raw_label_vs_rounded_pool_gap':((true_q*weights).sum((1,2))-y).abs()}


def clip_decoupled_gradients(model,maximum=1.):
    # A shared norm clip would indirectly couple classifier gradients to q.
    a=nn.utils.clip_grad_norm_(model.regressor.parameters(),maximum,error_if_nonfinite=True)
    b=nn.utils.clip_grad_norm_(model.classifier.parameters(),maximum,error_if_nonfinite=True)
    return a,b
