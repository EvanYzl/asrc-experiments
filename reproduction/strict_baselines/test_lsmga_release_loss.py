"""Compare current LSMGA loss/gradients with its frozen release source."""
import ast
import copy
import subprocess
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'reproduction/sources/LSMGA'
sys.path.insert(0,str(SOURCE))
from src.lsmga_model import LSMGA

released_source=subprocess.check_output(['git','-C',str(SOURCE),'show','HEAD:src/lsmga_model.py'],text=True)
tree=ast.parse(released_source)
model_class=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='LSMGA')
methods=[node for node in model_class.body if isinstance(node,ast.FunctionDef)
         and node.name in ['forward_kg','project_t','define_loss']]
namespace={'torch':torch}
exec(compile(ast.Module(body=methods,type_ignores=[]),'frozen_LSMGA_release','exec'),namespace)


class ReleasedLossTests(unittest.TestCase):
    def probe(self,released,relation_weights):
        obj=types.SimpleNamespace(device=torch.device('cpu'))
        obj.forward_GNN_embedding=lambda graph,index:graph
        obj.rel_embedding_layer=torch.nn.Embedding.from_pretrained(relation_weights.clone(),freeze=False)
        obj.project_t=types.MethodType(namespace['project_t'],obj)
        obj.define_loss=types.MethodType(namespace['define_loss'],obj)
        if released:
            # PyTorch 1.10 functional.py forwards these non-scalar tensors
            # directly to this native operator, allowing their broadcast.
            obj.criterion=lambda positive,negative,target:torch.margin_ranking_loss(positive,negative,target,.5,1)
        else:
            obj.criterion=torch.nn.MarginRankingLoss(margin=.5,reduction='mean')
            obj.released_ranking_loss=types.MethodType(LSMGA.released_ranking_loss,obj)
        return obj

    def test_published_forward_values_and_gradients(self):
        for batch in [1,2,7,200]:
            with self.subTest(batch=batch):
                torch.manual_seed(17000+batch)
                inputs=[torch.randn(batch,8,dtype=torch.float64) for _ in range(3)]
                weights=torch.randn(3,8,dtype=torch.float64)
                sample=torch.stack([torch.arange(batch),torch.arange(batch)%3,torch.arange(batch)],dim=1)
                outputs=[];gradients=[]
                for released in [True,False]:
                    model=self.probe(released,weights)
                    h,t,negative=[x.clone().requires_grad_(True) for x in inputs]
                    forward=namespace['forward_kg'] if released else LSMGA.forward_kg
                    loss=forward(model,h,sample,t,negative,0)
                    outputs.append(loss.detach())
                    gradients.append(torch.autograd.grad(loss,[h,t,negative,model.rel_embedding_layer.weight]))
                torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=1e-12)
                for expected,actual in zip(*gradients):
                    torch.testing.assert_close(expected,actual,rtol=0,atol=1e-12)

    def test_regression_example_distinguishes_broadcast_from_pairwise(self):
        model=types.SimpleNamespace(criterion=torch.nn.MarginRankingLoss(margin=.5))
        positive=torch.tensor([[.1],[.9]],dtype=torch.float64)
        negative=torch.tensor([.3,1.1],dtype=torch.float64)
        loss=LSMGA.released_ranking_loss(model,positive,negative)
        self.assertAlmostEqual(float(loss),.425)
        pairwise=torch.nn.functional.margin_ranking_loss(positive.flatten(),negative,-torch.ones_like(negative),margin=.5)
        self.assertAlmostEqual(float(pairwise),.3)
        self.assertNotAlmostEqual(float(loss),float(pairwise))


if __name__=='__main__':
    unittest.main()
