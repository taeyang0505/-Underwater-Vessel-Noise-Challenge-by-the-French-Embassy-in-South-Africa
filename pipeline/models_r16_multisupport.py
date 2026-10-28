"""Production identity for the independently checked relative17 backbones."""
from pipeline.models_r16_f1_preflight import R16F1PreflightModel
from pipeline.models_r16_production import decoupled_loss,clip_decoupled_gradients

class MultiSupportModel(R16F1PreflightModel):
 def architecture(self):
  a=super().architecture();a.pop('production_training_started')
  a.update(format='r16_multisupport_relative17_v1',architecture_version='16MS1',
           pretrained=False,requires_background_prior=False)
  return a
