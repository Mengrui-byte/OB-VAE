import hashlib,json,tempfile,unittest
from pathlib import Path
import torch
from obvae.api import ROOT,load_checkpoint
from model.autoencoder.direct_readout_vae import DirectReadoutOrderBookVAE
from model.layers.branch_precision import apply_branch_precision_
from train_dual.util.data import DualNormalizer

class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_export_strict_load_and_manifest(self):
        p=ROOT/'weights/gelu512.pt';expected=json.loads((ROOT/'weights/manifest.json').read_text())['gelu512.pt']
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),expected['sha256'])
        model,norm,saved=load_checkpoint(p)
        self.assertEqual(saved['step'],512);self.assertEqual(saved['source_steps'],25088)
        self.assertNotIn('optimizer',saved);self.assertEqual(sum(p.numel() for p in model.parameters()),20691748)
        self.assertEqual(norm.price_scale,104.2454906668734)
    def test_complete_256_minutes_shared_z_and_decode_only_at_32_levels(self):
        model=DirectReadoutOrderBookVAE(levels=32,quantity_scale=2.,activation_checkpointing=False,layout='frame1',frame_ffn=1536,use_shape=False,use_mass=False,readout_kernel='linear',readout_hidden=64).eval()
        apply_branch_precision_(model,encoder_fp32=True,decoders_fp32=True)
        t=torch.arange(256).float()[None,:,None,None];level=torch.arange(32).float()[None,None,None,:]
        price=torch.cat([10000+t*.1-level*.1,10001+t*.1+level*.1],dim=2)
        raw=dict(data=torch.stack([price,torch.ones_like(price)],-1),level_counts=torch.full((1,256,2),32,dtype=torch.int64))
        b=DualNormalizer(tick_size=.01).prepare(raw,'cpu');seen=[]
        hooks=[d.register_forward_pre_hook(lambda m,a:seen.append(a[0])) for d in [model.price_decoder,model.volume_decoder]]
        with torch.inference_mode():
            out=model(*[b[k] for k in ['price','volume','mask','tick_norm','times']],sample=False)
            replay=model.decode(out['z'],b['mask'],b['tick_norm'],b['times'])
        for h in hooks:h.remove()
        self.assertIs(seen[0],seen[1]);self.assertIs(seen[0],out['z'])
        self.assertEqual(tuple(out['z'].shape),(1,64,128));self.assertEqual(tuple(out['volume'].shape),(1,256,2,32))
        for k in ['price_norm','volume']:torch.testing.assert_close(out[k],replay[k],rtol=0,atol=0)
    def test_portable_snapshot(self):
        from util.model_sources import freeze_model_sources
        with tempfile.TemporaryDirectory() as temp:
            hashes=freeze_model_sources(temp,ROOT)
            self.assertIn('model/autoencoder/direct_readout_vae.py',hashes)

if __name__=='__main__':unittest.main()
