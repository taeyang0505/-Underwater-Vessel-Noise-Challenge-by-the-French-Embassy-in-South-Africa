"""Necessary policy tests: aggregate gain must never hide subgroup harm."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.multislice_gate_r19 import DEFAULT_POLICY, evaluate_candidate


class MultiSliceGateTests(unittest.TestCase):
    def setUp(self):
        self.panels={n:{'share':np.array([0.,.05,.5]),'active_minutes':np.array([0,2,12])} for n in DEFAULT_POLICY.panels}
        self.bases={n:np.array([.01,.05,.9]) for n in DEFAULT_POLICY.panels}

    def test_identity_passes(self):
        r=evaluate_candidate(self.panels,self.bases,{'identity':self.bases})['results']['identity']
        self.assertTrue(r['passed']);self.assertEqual(r['axes'],22)

    def test_aggregate_improvement_cannot_hide_weak_positive_harm(self):
        candidate={n:np.array([.01,.08,.5]) for n in DEFAULT_POLICY.panels}
        r=evaluate_candidate(self.panels,self.bases,{'harm':candidate})['results']['harm']
        self.assertFalse(r['passed'])
        self.assertTrue(all(c['delta']<0 for c in r['checks'] if c['slice']=='all'))
        self.assertTrue(any(c['slice']=='0<y<=0.10' and not c['passed'] for c in r['checks']))

    def test_missing_panel_rejects(self):
        candidate={n:v for n,v in self.bases.items() if n!='domain_holdout'}
        self.assertFalse(evaluate_candidate(self.panels,self.bases,{'missing':candidate})['results']['missing']['passed'])

    def test_nonfinite_prediction_rejects(self):
        candidate={n:v.copy() for n,v in self.bases.items()};candidate['local_gate'][1]=np.nan
        self.assertFalse(evaluate_candidate(self.panels,self.bases,{'nan':candidate})['results']['nan']['passed'])

    def test_empty_required_weak_slice_rejects(self):
        self.panels['local_gate']['share']=np.array([0.,.2,.5])
        self.assertFalse(evaluate_candidate(self.panels,self.bases,{'identity':self.bases})['results']['identity']['passed'])


if __name__=='__main__':unittest.main()
