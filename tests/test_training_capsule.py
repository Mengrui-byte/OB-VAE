import json,unittest
import torch
from obvae.api import ROOT,load_checkpoint
from train_dual.util.inherited_book_loss import inherited_book_loss
from train_dual.util.direct_readout_training import restore_direct_readout_adam

class TrainingCapsuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_saved_objective_builds_without_original_project(self):
        p=ROOT/'research';saved=torch.load(ROOT/'weights/gelu512.pt',map_location='cpu',mmap=True,weights_only=False)
        read=lambda name:json.loads((p/name).read_text())
        loss=inherited_book_loss(saved,read('plan.json'),'gelu',read('calibration.json'),read('quantity_calibration.json'),read('detail_calibration.json'))
        self.assertEqual(loss.relative_weight,.13141036760364427)
    @unittest.skipUnless((ROOT/'artifacts/gelu512_training.pt').exists(),'Full local training state not distributed in Git')
    def test_full_resume_optimizer_mapping_and_inference_export_exact(self):
        model,_,full=load_checkpoint(ROOT/'artifacts/gelu512_training.pt')
        export=torch.load(ROOT/'weights/gelu512.pt',map_location='cpu',mmap=True,weights_only=False)
        self.assertEqual(list(full['model']),list(export['model']))
        for k,v in full['model'].items():self.assertTrue(torch.equal(v,export['model'][k]),k)
        names=json.loads((ROOT/'research/plan.json').read_text())['original_parameter_names']
        optimizer,_=restore_direct_readout_adam(model,full['optimizer'],names)
        self.assertEqual(optimizer.state_dict()['param_groups'],full['optimizer']['param_groups'])
        self.assertEqual(len(optimizer.param_groups),4)

if __name__=='__main__':unittest.main()
