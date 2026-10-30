"""Meaningful policy/identity fixtures without production models or Test data."""
import csv,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from pipeline import r18_production_finalization as f

class FinalizationContractTests(unittest.TestCase):
    def test_pair_ids_do_not_merge_geometry_targets(self):
        # Matching IDs must not replace a panel's own target with other geometry.
        data={'fresh_fit_D0':{'db':np.zeros((2,15,25)),'share':np.array([0.,.1])},
              'fresh_fit_H0':{'db':np.zeros((2,15,25)),'share':np.array([0.,.6])}}
        def prediction(root,generation,family,name):
            val=np.array([.02,.12])if name.endswith('D0')else np.array([.02,.58])
            return{'share_D0':val,'share_H0':val,'prob_D0':np.array([.1,.9]),'prob_H0':np.array([.1,.9])}
        with patch.object(f,'dataset',side_effect=lambda root,name:data[name]),patch.object(f,'prediction',side_effect=prediction):
            s0,c0=f.blend('.', 'fresh_fit_D0','B1',1.)
            s1,c1=f.blend('.', 'fresh_fit_H0','B1',1.)
        np.testing.assert_allclose(s0,[.02,.12]);np.testing.assert_allclose(s1,[.02,.58])
        self.assertNotEqual(float(s0[1]),float(s1[1]))

    def test_fit_winner_gate_rejection_has_exact_old_fallback(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);folder=f.out(root);folder.mkdir(parents=True);f.write(folder/'PLAN.json',{})
            values=[]
            for label,b,a in f.candidates():
                for name in f.names('development'):
                    # New B2 wins fit, but gets worse on one gate axis.
                    value=.05 if label=='new_B2'else .1
                    if name in f.FIXED:value=.1 if label!='new_B2'else(.2 if name=='domain_holdout'else .09)
                    values.append({'panel':name,'candidate':label,'blend':b,'new_fraction':a,'rmse':value})
            with patch.object(f,'verify_plan'),patch.object(f,'completed_models',return_value={}),patch.object(f,'compare',return_value=([],values)):
                p=f.select(root)
            self.assertEqual(p['fit_winner']['candidate'],'new_B2')
            self.assertFalse(p['gate_adopted']);self.assertEqual(p['selected']['candidate'],'old_B1')
            before=(folder/'selected_policy.json').read_bytes()
            report_rows=[{'candidate':'old_B1','rmse':.8},{'candidate':'new_B2','rmse':.01}]
            with patch.object(f,'stage_guard'),patch.object(f,'compare',return_value=([],report_rows)):
                result=f.report(root)
            self.assertEqual(result['selected'],'old_B1')
            self.assertEqual(before,(folder/'selected_policy.json').read_bytes())

    def test_query_is_locked_before_any_raw_byte_open(self):
        with tempfile.TemporaryDirectory()as td,patch.object(Path,'read_bytes')as read_bytes:
            with self.assertRaises(FileNotFoundError):f.prepare_query(Path(td))
            read_bytes.assert_not_called()

    def test_single_query_capture_then_reuse_never_reopens_csv(self):
        from ops.predict_r16_production_nn import FEATURE_COLUMNS
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);folder=f.out(root);folder.mkdir(parents=True)
            f.write(folder/'DECISION_FREEZE.json',{})
            stream=io.StringIO();writer=csv.writer(stream);writer.writerow(['ID',*FEATURE_COLUMNS])
            writer.writerows([f'fixture_{i}',*([90.]*375)]for i in range(1500))
            p=root/'Test.csv';p.write_text(stream.getvalue())
            original=Path.read_bytes;opens=[]
            def capture(path):opens.append(str(path));return original(path)
            with patch.object(f,'stage_guard'),patch.object(Path,'read_bytes',capture):
                one=f.prepare_query(root,p);two=f.prepare_query(root,p)
            self.assertEqual(opens,[str(p)])
            self.assertEqual(one,two);self.assertEqual(one['raw_Test_open_count'],1)
            self.assertFalse(one['Test_used_for_fit'])

if __name__=='__main__':unittest.main()
