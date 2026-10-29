import unittest
import numpy as np
import torch

from model.losses.quantity_strata import stratified_quantity_huber, StratifiedQuantityBookLoss
from model.losses.recall_guard import RecallGuardBookLoss
from train_dual.util.quantity_strata import QuantityStrataMetrics


class QuantityStrataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_bin_prevalence_does_not_dilute_error_in_one_quantity_scale(self):
        scales = [.005,.05,.5,5.,50.,500.]
        truth = torch.tensor(scales).reshape(1,1,1,6)
        pred = truth.clone(); pred[...,0] *= 2
        one,_,_ = stratified_quantity_huber(pred,truth,torch.ones_like(truth,dtype=torch.bool),scales)
        many_q = torch.cat([truth,truth[...,:1].expand(1,1,1,1000)],-1)
        many_p = torch.cat([pred,pred[...,:1].expand(1,1,1,1000)],-1)
        many,_,_ = stratified_quantity_huber(many_p,many_q,torch.ones_like(many_q,dtype=torch.bool),scales)
        torch.testing.assert_close(one,torch.tensor(.5/6))
        torch.testing.assert_close(one,many,rtol=0,atol=0)

    def test_small_and_large_gradients_padding_and_empty_windows(self):
        truth = torch.tensor([.005,500.,float('nan')]).reshape(1,1,1,3)
        pred = torch.tensor([.01,250.,float('nan')]).reshape(1,1,1,3).requires_grad_()
        mask = torch.tensor([True,True,False]).reshape_as(truth)
        loss,_,counts = stratified_quantity_huber(pred,truth,mask,[.005,.05,.5,5,50,500])
        loss.backward(); self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(pred.grad[...,0]),0.); self.assertLess(float(pred.grad[...,1]),0.)
        self.assertEqual(float(pred.grad[...,2]),0.); self.assertEqual(int(counts.sum()),2)
        pred.grad = None
        zero,_,_ = stratified_quantity_huber(pred,truth,torch.zeros_like(mask),[.005,.05,.5,5,50,500])
        zero.backward(); self.assertEqual(float(zero.detach()),0.); self.assertEqual(float(pred.grad.abs().sum()),0.)

    def test_zero_coefficient_preserves_original_loss_and_gradient(self):
        target = torch.ones(1,2,2,9); target[...,4]=150
        a=torch.full_like(target,.4,requires_grad=True); b=a.detach().clone().requires_grad_()
        batch=dict(volume=target,price=torch.zeros_like(target),mask=torch.ones_like(target,dtype=torch.bool))
        def output(q):return dict(volume=q,price_norm=torch.zeros_like(q),mid_norm=torch.zeros(1,2),mu=torch.zeros(1,64,128),logvar=torch.zeros(1,64,128))
        kw=dict(mode='shape_mass',use_tail=True,tail_weight=.04,protect_mid=True,mid_weight=.8,mid_delta_weight=64.)
        old=RecallGuardBookLoss(2.,**kw)(output(a),batch)
        new=StratifiedQuantityBookLoss(2.,bin_scales=[.005,.05,.5,5,50,500],stratified_weight=0.,**kw)(output(b),batch)
        torch.testing.assert_close(old['loss'],new['loss'],rtol=0,atol=0)
        old['loss'].backward();new['loss'].backward();torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

    def test_streaming_bin_metrics_are_exact_for_perfect_predictions(self):
        q=np.array([.001,.05,.5,5.,50.,500.]); mask=np.ones(6,dtype=bool)
        metrics=QuantityStrataMetrics();metrics.update(q,q,mask);metrics.update(q,q,mask)
        rows=metrics.finish()
        for row in rows:
            self.assertEqual(row['count'],2);self.assertEqual(row['RMSE_BTC'],0.)
            self.assertEqual(row['within20percent'],1.);self.assertEqual(row['quantity_mass_ratio'],1.)
        empty=QuantityStrataMetrics().finish();self.assertTrue(all(row['RMSE_BTC'] is None for row in empty))
        with self.assertRaises(ValueError):stratified_quantity_huber(torch.ones(1,2),torch.ones(1,2),torch.ones(1,2,dtype=torch.bool),[0,1,1,1,1,1])

    def test_decimal_boundaries_match_torch_loss_and_scale_fit(self):
        from train_dual.util.quantity_strata import quantity_bin_mask_numpy
        q=torch.tensor([0,.001,.009,.010,.011,.099,.100,.101,.999,1.,9.999,10.,99.999,100.]).reshape(1,1,1,-1)
        mask=torch.ones_like(q,dtype=torch.bool)
        _,_,counts=stratified_quantity_huber(q,q,mask,[.005,.05,.5,5,50,500])
        metrics=QuantityStrataMetrics();metrics.update(q.double().numpy(),q.double().numpy(),mask.numpy())
        self.assertEqual(counts.tolist(),[row['count'] for row in metrics.finish()])
        self.assertEqual(counts.tolist(),[int(quantity_bin_mask_numpy(q.double().numpy(),i).sum()) for i in range(6)])
        self.assertFalse(bool(quantity_bin_mask_numpy(np.array([np.float32(.01)],dtype=np.float64),0)[0]))
        self.assertTrue(bool(quantity_bin_mask_numpy(np.array([np.float32(.01)],dtype=np.float64),1)[0]))

    def test_streaming_wrapper_preserves_all_original_metrics(self):
        from model.autoencoder.local_hint_vae import LocalHintOrderBookVAE
        from tests.fixtures import raw_batch
        from train_dual.util.data import DualNormalizer
        from train_dual.experiments.long_worker import evaluate
        from train_dual.util.quantity_strata import evaluate_with_quantity_strata
        raw=raw_batch(size=1,frames=256,levels=128)
        data=[{k:v[0] for k,v in raw.items()}]
        model=LocalHintOrderBookVAE(levels=128,quantity_scale=2.,activation_checkpointing=False)
        norm=DualNormalizer(tick_size=.25)
        expected=evaluate(model,norm,data,1,'cpu')
        extended=evaluate_with_quantity_strata(model,norm,data,1,'cpu')
        strata=extended.pop('quantity_strata')
        self.assertEqual(extended.pop('quantity_strata_schema_version'),2)
        self.assertIn('float32',extended.pop('quantity_strata_membership'))
        self.assertEqual(expected,extended)
        self.assertEqual(sum(row['count'] for row in strata),int(norm.prepare(raw,'cpu')['mask'].sum()))


if __name__=='__main__':unittest.main()
