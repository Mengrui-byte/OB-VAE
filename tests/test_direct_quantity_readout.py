import unittest
from copy import deepcopy
from pathlib import Path
import tempfile

import torch

from model.layers.direct_quantity_readout import FrameQuantityLogitMLP


class DirectQuantityReadoutTests(unittest.TestCase):
    def test_matched_parameters_zero_residual_and_rng_preservation(self):
        torch.manual_seed(42)
        before = torch.get_rng_state().clone()
        affine = FrameQuantityLogitMLP(levels=16, frame_dim=8, hidden_dim=32, activation='identity')
        nonlinear = FrameQuantityLogitMLP(levels=16, frame_dim=8, hidden_dim=32, activation='gelu')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(list(affine.state_dict()), list(nonlinear.state_dict()))
        for name, value in affine.state_dict().items(): self.assertTrue(torch.equal(value, nonlinear.state_dict()[name]))
        frame = torch.randn(2, 3, 8)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            a, b = affine(frame), nonlinear(frame)
        self.assertEqual(a.dtype, torch.float32)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(tuple(a.shape), (2, 3, 2, 16))
        self.assertEqual(a.count_nonzero(), 0)

    def test_independent_single_level_output_and_gradient_flow(self):
        torch.set_num_threads(2)
        model = FrameQuantityLogitMLP(levels=16, frame_dim=8, hidden_dim=32)
        with torch.no_grad(): model.output_projection.bias[19] = 1.
        output = model(torch.randn(1, 2, 8))
        self.assertTrue(torch.equal(output[0, :, 1, 3], torch.ones(2)))
        self.assertEqual(int(output.count_nonzero()), 2)
        with torch.no_grad(): model.output_projection.bias.zero_()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        frame, target = torch.randn(2, 3, 8), torch.randn(2, 3, 2, 16)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(frame) - target).square().mean()
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(float(model.output_projection.weight.grad.norm()), 0.)
            if step == 0: self.assertEqual(float(model.input_projection.weight.grad.norm()), 0.)
            else: self.assertGreater(float(model.input_projection.weight.grad.norm()), 0.)
            optimizer.step()

    def test_complete256frame_singleZ_source_parity_on_small_level_fixture(self):
        from model.autoencoder.regional_phase_vae import RegionalPhaseOrderBookVAE
        from model.autoencoder.direct_readout_vae import DirectReadoutOrderBookVAE
        from model.layers.branch_precision import apply_branch_precision_
        torch.set_num_threads(2)
        settings = dict(levels=32, quantity_scale=2., activation_checkpointing=False,
                        layout='frame1', use_shape=False, use_mass=False, readout_kernel='linear')
        torch.manual_seed(42)
        source = RegionalPhaseOrderBookVAE(**settings).eval()
        after_source = torch.get_rng_state().clone()
        torch.manual_seed(42)
        model = DirectReadoutOrderBookVAE(readout_hidden=64, **settings).eval()
        self.assertTrue(torch.equal(after_source, torch.get_rng_state()))
        source_names = [n for n, _ in source.named_parameters()]
        candidate_names = [n for n, _ in model.named_parameters() if not n.startswith('direct_quantity_readout.')]
        self.assertEqual(source_names, candidate_names)
        for name, value in source.state_dict().items(): self.assertTrue(torch.equal(value, model.state_dict()[name]))
        apply_branch_precision_(model, encoder_fp32=True, decoders_fp32=True)
        z = torch.randn(1, 64, 128)
        mask = torch.ones(1, 256, 2, 32, dtype=torch.bool); mask[:, :, :, -1] = False
        times = torch.arange(256)[None].float(); tick = torch.tensor([.001])
        received = []
        hooks = [d.register_forward_pre_hook(lambda m, args: received.append(args[0]))
                 for d in [model.price_decoder, model.volume_decoder]]
        with torch.inference_mode():
            before, after = source.decode(z, mask, tick, times), model.decode(z, mask, tick, times)
        for hook in hooks: hook.remove()
        self.assertTrue(received[0] is received[1] is z)
        for key in ['price_norm', 'mid_norm', 'gap_norm', 'volume', 'quantity_logit', 'hint_local']:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertEqual(tuple(after['volume'].shape), (1, 256, 2, 32))
        self.assertEqual(int(after['direct_quantity_logit_delta'].count_nonzero()), 0)

    def test_checkpoint_roundtrip_and_audited_admission_schema(self):
        from model.autoencoder.direct_readout_vae import DirectReadoutOrderBookVAE
        from model.layers.branch_precision import apply_branch_precision_, branch_precision_policy
        from train_dual.util.checkpoint import load_for_inference
        from obvae.api import admit_neural_payload
        torch.set_num_threads(2)
        regional = dict(layout='frame1', use_shape=False, use_mass=False, readout_kernel='linear')
        unified = dict(levels=32, quantity_scale=2., activation_checkpointing=False)
        model = DirectReadoutOrderBookVAE(readout_hidden=64, **regional, **unified).eval()
        apply_branch_precision_(model, encoder_fp32=True, decoders_fp32=True)
        with torch.no_grad(): model.direct_quantity_readout.output_projection.bias[19] = .1
        payload = dict(format_version=1, model=model.state_dict(), unified_vae_config=unified,
            regional_phase_config=regional, direct_readout_config=model.direct_readout_config,
            branch_precision_config=branch_precision_policy(model), config=dict(normalization=dict(tick_size=.01, price_scale=50.)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'direct.pt'; torch.save(payload, path)
            fresh, _ = load_for_inference(path)
        self.assertEqual(fresh.direct_readout_config, model.direct_readout_config)
        z = torch.randn(1, 64, 128); mask = torch.ones(1, 256, 2, 32, dtype=torch.bool)
        tick, times = torch.tensor([.001]), torch.arange(256)[None].float()
        def forbidden_encoder(*args): raise AssertionError('Fresh Z-only decode touched Encoder')
        hook = fresh.encoder.register_forward_pre_hook(forbidden_encoder)
        with torch.inference_mode(): a, b = model.decode(z, mask, tick, times), fresh.decode(z, mask, tick, times)
        hook.remove()
        for key in ['price_norm', 'volume', 'quantity_logit', 'direct_quantity_logit_delta']:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, '2048-level'): admit_neural_payload(payload)
        schema = dict(format_version=1, unified_vae_config=dict(levels=2048,latent_channels=128),
            regional_phase_config=regional, branch_precision_config=dict(encoder_fp32=True,decoders_fp32=True),
            direct_readout_config=dict(readout_activation='gelu',readout_hidden=2048,readout_seed=42))
        self.assertEqual(admit_neural_payload(schema),'direct_readout')
        for name,value in [('readout_activation','relu'),('readout_hidden',1024),('readout_seed',43)]:
            bad=deepcopy(schema);bad['direct_readout_config'][name]=value
            with self.assertRaises(ValueError):admit_neural_payload(bad)
        bad=deepcopy(schema);bad['branch_precision_config']['encoder_fp32']=False
        with self.assertRaisesRegex(ValueError,'fullFP32'):admit_neural_payload(bad)
        bad=deepcopy(schema);bad['paired_logit_config']=dict(layout='frame1',phase_specific=True)
        with self.assertRaisesRegex(ValueError,'combination'):admit_neural_payload(bad)

    def test_three_Adam_groups_preserved_and_new_group_starts_empty(self):
        from model.autoencoder.direct_readout_vae import DirectReadoutOrderBookVAE
        from train_dual.util.direct_readout_training import restore_direct_readout_adam
        torch.set_num_threads(2)
        model = DirectReadoutOrderBookVAE(levels=32, quantity_scale=2., activation_checkpointing=False,
            layout='frame1', readout_kernel='linear', readout_hidden=64)
        named = dict(model.named_parameters())
        names = [[n for n in named if not n.startswith(('hint_adapter.', 'local_readout.', 'direct_quantity_readout.'))],
                 [n for n in named if n.startswith('hint_adapter.')], [n for n in named if n.startswith('local_readout.')]]
        old = torch.optim.AdamW([dict(params=[named[n] for n in group]) for group in names], lr=1e-4)
        for group in names: named[group[0]].grad = torch.ones_like(named[group[0]])
        old.step(); old.zero_grad(set_to_none=True)
        saved = deepcopy(old.state_dict())
        fresh, groups = restore_direct_readout_adam(model, saved, names[0])
        self.assertEqual(fresh.state_dict()['param_groups'][:3], saved['param_groups'])
        for index, state in saved['state'].items():
            for key, value in state.items(): torch.testing.assert_close(fresh.state_dict()['state'][index][key], value, rtol=0, atol=0)
        self.assertEqual(len(groups), 4)
        self.assertTrue(all(named[n] not in fresh.state for n in groups[3]))
        named[groups[3][-1]].grad = torch.ones_like(named[groups[3][-1]])
        fresh.step()
        for index, state in saved['state'].items():
            for key, value in state.items(): torch.testing.assert_close(fresh.state_dict()['state'][index][key], value, rtol=0, atol=0)
        resumed = deepcopy(fresh.state_dict())
        again, _ = restore_direct_readout_adam(model, resumed, names[0])
        self.assertEqual(again.state_dict()['param_groups'], resumed['param_groups'])
        with self.assertRaisesRegex(ValueError, 'mapping'): restore_direct_readout_adam(model, saved, list(reversed(names[0])))


if __name__ == '__main__': unittest.main()
