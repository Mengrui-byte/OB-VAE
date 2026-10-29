"""Prevent fabricated cross-split, cross-checkpoint and terminal-only selections."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools import build_branch_atlas as atlas


def result(quantity=2.,windows=16):
    return dict(result=dict(windows=windows,metrics=dict(price_usdt=dict(rmse=10.),quantity_btc=dict(rmse=quantity,count=16777216,target_rms=2.2)),thresholds={'100':dict(tp=4,fp=2,fn=4,recall=.5,precision=2/3)}))

class BranchAtlasTests(unittest.TestCase):
    def test_original_count_based_f1_is_preserved(self):
        a=atlas.normalized(result()['result']);self.assertAlmostEqual(a['F1'],8/14)
        self.assertEqual(a['windows'],16)
    def test_synthetic_pilot_does_not_enter_test_ranking(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for p,q in [('run/lane/step128_evaluation.json',2.),('run/pilot/lane/step128_evaluation.json',.001)]:
                path=root/p;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result(q)))
            with patch.object(atlas,'SOURCE',root):rows=atlas.scan(dict(id='frame1',experiments=['run']))
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['quantity'],2.)
            self.assertNotIn('/pilot/',rows[0]['path'])
    def test_best_intermediate_is_retained_and_common_endpoint_marked(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'run/lane').mkdir(parents=True)
            for step,q in [(128,1.),(256,2.)]:(root/f'run/lane/step{step}_evaluation.json').write_text(json.dumps(result(q)))
            with patch.object(atlas,'SOURCE',root):rows=atlas.scan(dict(id='frame1',experiments=['run']))
            self.assertEqual({r['step'] for r in rows},{128,256})
            self.assertEqual(min(rows,key=lambda r:r['quantity'])['step'],128)
            self.assertEqual([r['step'] for r in rows if r['is_common_endpoint']],[256])
    def test_later_weight_is_not_substituted_for_missing_best_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);g=root/'run/lane';g.mkdir(parents=True)
            p=g/'step128_evaluation.json';p.write_text(json.dumps(result(1.)))
            (g/'final_evaluation.json').write_text(json.dumps(result(2.)));(g/'final.pt').touch()
            with patch.object(atlas,'SOURCE',root):self.assertIsNone(atlas.weight_for(p,'',128))
    def test_seen_and_training_files_are_not_test_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);g=root/'run/lane';g.mkdir(parents=True)
            for name in ['step128_evaluation.json','step128_seen_evaluation.json','step128_seen.json','train_step128_16windows.json']:
                (g/name).write_text(json.dumps(result()))
            with patch.object(atlas,'SOURCE',root):rows=atlas.scan(dict(id='frame1',experiments=['run']))
            self.assertEqual(len(rows),1);self.assertTrue(rows[0]['path'].endswith('step128_evaluation.json'))
    def test_every_documented_branch_has_an_architecture(self):
        self.assertEqual({n['id'] for n in atlas.NODES},set(atlas.ARCH))

if __name__=='__main__':unittest.main()
