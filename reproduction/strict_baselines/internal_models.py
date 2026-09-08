"""Shared, frozen TransE interface for Target-only and internal transfer controls."""
from common import *
from internal_data import CSRStore,NegativeSampler,alignments,load_setting,source_kgs
import torch.nn as nn
import torch.nn.functional as F

DIM=128


class TargetModel(nn.Module):
    def __init__(self,nentity,nrelation):
        super().__init__();self.entity=nn.Embedding(nentity,DIM);self.relation=nn.Embedding(nrelation,DIM)
        nn.init.uniform_(self.entity.weight,-6/DIM**.5,6/DIM**.5)
        nn.init.uniform_(self.relation.weight,-6/DIM**.5,6/DIM**.5)
        self.register_buffer('_normalized_table',None,persistent=False)

    def entities(self):
        if self.entity.weight.requires_grad:return F.normalize(self.entity.weight,p=2,dim=-1)
        if self._normalized_table is None:self._normalized_table=F.normalize(self.entity.weight.detach(),p=2,dim=-1)
        return self._normalized_table
    def query(self,triples):return self.entities()[triples[:,0]]+self.relation(triples[:,1])
    def all_scores(self,triples,offset=None):
        query=self.query(triples)
        if offset is not None:query=query+offset
        return -torch.cdist(query,self.entities(),p=2)
    def sampled_scores(self,triples,negative,offset=None):
        query=self.query(triples)
        if offset is not None:query=query+offset
        ent=self.entities();positive=-(query-ent[triples[:,2]]).norm(p=2,dim=-1)
        neg=-(query[:,None,:]-ent[negative]).norm(p=2,dim=-1)
        return positive,neg


def load_target(dataset,kg,root):
    d=load_setting(dataset,kg);model=TargetModel(d['entities'],d['relations'])
    path=Path(root)/kg/'best.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);return model


class FusionModel(nn.Module):
    def __init__(self,dataset,target,base_root,target_base_root=None,condition='full'):
        super().__init__();self.dataset=dataset;self.target=target
        self.base=load_target(dataset,target,target_base_root or base_root)
        self.sources=[s for s in source_kgs(dataset) if s!=target]
        self.source_models=nn.ModuleDict({s:load_target(dataset,s,base_root) for s in self.sources})
        for parameter in self.parameters():parameter.requires_grad_(False)
        self.edge_encoder=nn.Linear(DIM,DIM,bias=True);self.direction=nn.Embedding(2,DIM)
        self.projections=nn.ModuleDict({s:nn.Linear(DIM,DIM,bias=False) for s in self.sources})
        for projection in self.projections.values():nn.init.zeros_(projection.weight)
        self.norm=nn.LayerNorm(DIM,elementwise_affine=False)
        self.stores={s:CSRStore(dataset,s) for s in self.sources};self.set_condition(condition)
        self.gamma=1.

    def set_condition(self,condition):
        self.condition=condition;self.mapping=alignments(self.dataset,self.target,condition if condition!='target20' else 'full')

    def availability(self,heads):
        heads=np.asarray(heads,dtype=np.int64)
        aligned=np.stack([self.mapping[s][heads] for s in self.sources],axis=1)
        degree=np.stack([np.where(aligned[:,i]>=0,self.stores[s].degree[np.maximum(aligned[:,i],0)],0) for i,s in enumerate(self.sources)],axis=1)
        return aligned,(aligned>=0)&(degree>0),np.minimum(degree,32).astype(np.int32)

    def messages(self,heads,admitted=None,save_edges=False):
        heads=np.asarray(heads,dtype=np.int64);aligned,available,counts=self.availability(heads)
        if admitted is not None:available&=np.asarray(admitted,dtype=bool)
        device=self.base.entity.weight.device;messages=[];records={}
        for i,s in enumerate(self.sources):
            # Rejected sources never enter the CSR fetch path.
            rows=np.flatnonzero(available[:,i]);message=torch.zeros((len(heads),DIM),device=device)
            if len(rows):
                raw=self.stores[s].fetch(aligned[rows,i]);src=self.source_models[s]
                neighbor=torch.as_tensor(raw['neighbor'],device=device);relation=torch.as_tensor(raw['relation'],device=device)
                direction=torch.as_tensor(raw['direction'],device=device);valid=torch.as_tensor(raw['valid'],device=device)
                encoded=torch.tanh(self.edge_encoder(src.entities()[neighbor]+src.relation(relation)+self.direction(direction)))
                mean=(encoded*valid[:,:,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
                message=message.index_copy(0,torch.as_tensor(rows,device=device),self.projections[s](self.norm(mean)))
                if save_edges:records[s]={'rows':rows,'source_entities':aligned[rows,i],**raw}
            messages.append(message)
        return torch.stack(messages,dim=1),torch.as_tensor(available,device=device),counts*available,records

    def weights(self,mask,mode='uniform',attention=None,features=None):
        if mode=='attention':
            logits=attention(features).squeeze(-1).masked_fill(~mask,-1e30)
            weights=logits.softmax(dim=1)*mask
        else:weights=mask.float()
        return weights/weights.sum(dim=1,keepdim=True).clamp_min(1e-12)

    def offset(self,triples,mode='uniform',attention=None,admitted=None):
        heads=triples[:,0].detach().cpu().numpy()
        messages,mask,counts,records=self.messages(heads,admitted)
        features=self.features(triples) if mode=='attention' else None
        weights=self.weights(mask,mode,attention,features)
        return self.gamma*(messages*weights[:,:,None]).sum(1),mask,weights,counts

    def all_scores_numpy(self,triples,mode='uniform',attention=None,admitted=None):
        t=torch.as_tensor(triples.copy(),device=self.base.entity.weight.device)
        if mode=='target':return self.base.all_scores(t)
        offset,mask,weights,counts=self.offset(t,mode,attention,admitted)
        return self.base.all_scores(t,offset)

    def raw_features(self,triples):
        heads=triples[:,0].detach().cpu().numpy();aligned,available,counts=self.availability(heads)
        sketches=[]
        for i,s in enumerate(self.sources):
            sk=np.array(self.stores[s].sketch[np.maximum(aligned[:,i],0)],dtype=np.float32)
            sk[~available[:,i]]=0;sketches.append(np.log1p(sk))
        current=torch.as_tensor(np.stack(sketches,axis=1),device=triples.device)
        mask=torch.as_tensor(available,device=triples.device);other_n=mask.sum(1,keepdim=True)-mask.int()
        other=(current.sum(1,keepdim=True)-current)/other_n.clamp_min(1)[:,:,None]
        query=torch.cat([self.base.entities()[triples[:,0]],self.base.relation(triples[:,1])],dim=-1)
        return torch.cat([query[:,None,:].expand(-1,len(self.sources),-1),current,other,(other_n==0).float()[:,:,None]],dim=-1)

    def features(self,triples):return (self.raw_features(triples)-self.feature_mean)/self.feature_std


class QueryAttention(nn.Module):
    def __init__(self):
        super().__init__();self.network=nn.Sequential(nn.Linear(2*DIM+1024+1,256),nn.GELU(),nn.Linear(256,1))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
    def forward(self,x):return self.network(x)


def sampled_loss(model,triples,negative,offset=None):
    pos,neg=model.sampled_scores(triples,negative,offset)
    return F.relu(.5-pos[:,None]+neg).mean()
