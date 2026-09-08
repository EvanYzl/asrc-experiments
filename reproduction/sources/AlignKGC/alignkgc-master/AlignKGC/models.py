import numpy as np
import math
import torch
import torch.nn as nn


class distmult(nn.Module):
    """
    DistMult Model from Trullion et al 2014.\n
    Scoring function (s, r, o) = <s, r, o> # dot product
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False, reg=2, batch_norm=False):
        """
        The initializing function. These parameters are expected to be supplied from the command line when running the\n
        program from main.\n
        :param entity_count: The number of entities in the knowledge base/model
        :param relation_count: Number of relations in the knowledge base/model
        :param embedding_dim: The size of the embeddings of entities and relations
        :param unit_reg: Whether the ___entity___ embeddings should be unit regularized or not
        :param clamp_v: The value at which to clamp the scores. (necessary to avoid over/underflow with some losses
        :param display_norms: Whether to display the max and min entity and relation embedding norms with each update
        :param reg: The type of regularization (example-l1,l2)
        """
        super(distmult, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.reg = reg

        self.display_norms = display_norms
        self.E = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R = nn.Embedding(self.relation_count, self.embedding_dim)
        nn.init.normal_(self.E.weight.data, 0, 0.05)
        nn.init.normal_(self.R.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        self.batch_norm = batch_norm
        if batch_norm:
            E_bn = nn.BatchNorm1d(self.embedding_dim)
            R_bn = nn.BatchNorm1d(self.embedding_dim)


    def forward(self, s, r, o, flag_debug=0, using_sm=0):
        """
        This is the scoring function \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the scoring function
        """
        s_e = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r_e = self.R(r)
        o_e = self.E(o) if o is not None else self.E.weight.unsqueeze(0)

        if self.batch_norm:
            s_e = self.E_bn(s_e)
            o_e = self.E_bn(o_e)
            r_e = self.R_bn(r_e)

        if self.clamp_v:
            s_e.data.clamp_(-self.clamp_v, self.clamp_v)
            r_e.data.clamp_(-self.clamp_v, self.clamp_v)
            o_e.data.clamp_(-self.clamp_v, self.clamp_v)
        #'''
        if o is None or o.shape[1] > 1: 
            '''
            tmp1 = torch.einsum('ijk,ijk->ijk', [s_e,r_e]); #print("tmp1 bef:",tmp1.shape);#tmp1 = tmp1.view(-1,self.embedding_dim); print("tmp1 after:",tmp1.shape)
            result = torch.einsum('ijk,ilk->il',[tmp1, o_e])#tmp1 @ s_e
            '''
            '''
            tmp1 = s_e * r_e
            result = (tmp1 * o_e).sum(dim=-1)
            '''
            tmp1 = s_e * r_e
            if o is not None:
                o_e = o_e[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_e
            else:
                o_e = o_e.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_e
            result = result.squeeze(1)
        else:
            '''
            elif s is None or s.shape[1] > 1:
                tmp1 = torch.einsum('ijk,ijk->ijk', [o_e,r_e]);  #print("tmp1 bef:",tmp1.shape); #tmp1 = tmp1.view(-1,self.embedding_dim);  print("tmp1 afer:",tmp1.shape)
                result = torch.einsum('ijk,ilk->il',[tmp1, s_e])#tmp1 @ s_e
            '''
            '''
            #else:
            tmp1 = o_e * r_e
            result = (tmp1 * s_e).sum(dim=-1)
            '''
            '''
            tmp1 = torch.einsum('ijk,ijk->ijk', [o_e,r_e]);  #print("tmp1 bef:",tmp1.shape); #tmp1 = tmp1.view(-1,self.embedding_dim);  print("tmp1 afer:",tmp1.shape)
            result = torch.einsum('ijk,ijk->i', [tmp1,s_e]);  
            result = result.unsqueeze(-1)
            '''
            tmp1 = o_e * r_e
            if o is not None:
                o_e = o_e[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_e
            else:
                s_e = s_e.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_e
            result = result.squeeze(1)
        return result#(s_e*r_e*o_e).sum(dim=-1)#result#(s*r*o).sum(dim=-1)


    def regularizer(self, s, r, o):
        """
        This is the regularization term \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the regularization term
        """
        s = self.E(s)
        r = self.R(r)
        o = self.E(o)
        if self.reg==2:
            return (s*s+o*o+r*r).sum()
        elif self.reg == 1:
            return (s.abs()+r.abs()+o.abs()).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s**2),torch.sqrt(o**2),torch.sqrt(r**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg/s.shape[0]
        else:
            print("Unknown reg for distmult model")
            assert(False)

    def post_epoch(self):
        """
        Post epoch/batch processing stuff.
        :return: Any message that needs to be displayed after each batch
        """
        '''
        if (not self.unit_reg and not self.display_norms):
            return ""
        e_norms = self.E.weight.data.norm(2, dim=-1)
        r_norms = self.R.weight.data.norm(2, dim=-1)
        max_e, min_e = torch.max(e_norms), torch.min(e_norms)
        max_r, min_r = torch.max(r_norms), torch.min(r_norms)
        if self.unit_reg:
            self.E.weight.data.div_(e_norms.unsqueeze(1))
        if self.display_norms:
            return "E[%4f, %4f] R[%4f, %4f]" % (max_e, min_e, max_r, min_r)
        else:
            return ""
        '''
        if(self.unit_reg):
            self.E.weight.data.div_(self.E.weight.data.norm(2, dim=-1, keepdim=True))
            self.R.weight.data.div_(self.R.weight.data.norm(2, dim=-1, keepdim=True))
        return ""


class SimplIE_v1(nn.Module):
    """
    DistMult Model from Trullion et al 2014.\n
    Scoring function (s, r, o) = <s, r, o> # dot product
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False, reg=2, batch_norm=False, flag_add_reverse=1, flag_avg_scores=0):
        """
        The initializing function. These parameters are expected to be supplied from the command line when running the\n
        program from main.\n
        :param entity_count: The number of entities in the knowledge base/model
        :param relation_count: Number of relations in the knowledge base/model
        :param embedding_dim: The size of the embeddings of entities and relations
        :param unit_reg: Whether the ___entity___ embeddings should be unit regularized or not
        :param clamp_v: The value at which to clamp the scores. (necessary to avoid over/underflow with some losses
        :param display_norms: Whether to display the max and min entity and relation embedding norms with each update
        :param reg: The type of regularization (example-l1,l2)
        """
        super(SimplIE_v1, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.reg = reg

        self.display_norms = display_norms
        self.E_s = nn.Embedding(self.entity_count, self.embedding_dim)
        self.E_o = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R = nn.Embedding(self.relation_count, self.embedding_dim)
        nn.init.normal_(self.E_s.weight.data, 0, 0.05)
        nn.init.normal_(self.E_o.weight.data, 0, 0.05)
        nn.init.normal_(self.R.weight.data, 0, 0.05)

        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        self.batch_norm = batch_norm
        if batch_norm:
            E_s_bn = nn.BatchNorm1d(self.embedding_dim)
            E_o_bn = nn.BatchNorm1d(self.embedding_dim)
            R_bn = nn.BatchNorm1d(self.embedding_dim)
        if flag_add_reverse:
            print("CX: Added inverse relations!")
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count

        self.flag_add_reverse = flag_add_reverse
        self.flag_avg_scores  = flag_avg_scores

    def forward(self, s, r, o, flag_debug=0):
        """
        This is the scoring function \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the scoring function
        """
        s_e = self.E_s(s) if s is not None else self.E_s.weight.unsqueeze(0)
        r_e = self.R(r)
        o_e = self.E_o(o) if o is not None else self.E_o.weight.unsqueeze(0)

        #if s is None:
        #    pdb.set_trace()

        if self.batch_norm:
            s_e = self.E_s_bn(s_e)
            o_e = self.E_o_bn(o_e)
            r_e = self.R_bn(r_e)

        if self.clamp_v:
            s_e.data.clamp_(-self.clamp_v, self.clamp_v)
            r_e.data.clamp_(-self.clamp_v, self.clamp_v)
            o_e.data.clamp_(-self.clamp_v, self.clamp_v)
        #if o is not None:
        #    if o.shape[0] == 10:
        #        pdb.set_trace()
        if o is None or o.shape[1] > 1:
            tmp1 = s_e * r_e
            if o is not None:
                o_e = o_e[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_e
            else:
                o_e = o_e.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_e
            result = result.squeeze(1)
        else:
            tmp1 = o_e * r_e
            if s is not None:
                s_e = s_e[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_e
            else:
                s_e = s_e.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_e
            result = result.squeeze(1)
        return result


    def regularizer(self, s, r, o):
        """
        This is the regularization term \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the regularization term
        """
        s = self.E_s(s)
        r = self.R(r)
        o = self.E_o(o)
        if self.reg==2:
            return (s*s+o*o+r*r).sum()
        elif self.reg == 1:
            return (s.abs()+r.abs()+o.abs()).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s**2),torch.sqrt(o**2),torch.sqrt(r**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg/s.shape[0]
        else:
            print("Unknown reg for distmult model")
            assert(False)

    def post_epoch(self):
        """
        Post epoch/batch processing stuff.
        :return: Any message that needs to be displayed after each batch
        """
        if(self.unit_reg):
            self.E_s.weight.data.div_(self.E.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_o.weight.data.div_(self.E.weight.data.norm(2, dim=-1, keepdim=True))
            self.R.weight.data.div_(self.R.weight.data.norm(2, dim=-1, keepdim=True))
        return ""

class SimplIE_v2(nn.Module):
    """
    DistMult Model from Trullion et al 2014.\n
    Scoring function (s, r, o) = <s, r, o> # dot product
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False, reg=2, batch_norm=False, flag_add_reverse=0, flag_avg_scores=0):
        """
        The initializing function. These parameters are expected to be supplied from the command line when running the\n
        program from main.\n
        :param entity_count: The number of entities in the knowledge base/model
        :param relation_count: Number of relations in the knowledge base/model
        :param embedding_dim: The size of the embeddings of entities and relations
        :param unit_reg: Whether the ___entity___ embeddings should be unit regularized or not
        :param clamp_v: The value at which to clamp the scores. (necessary to avoid over/underflow with some losses
        :param display_norms: Whether to display the max and min entity and relation embedding norms with each update
        :param reg: The type of regularization (example-l1,l2)
        """
        super(SimplIE_v2, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.reg = reg

        self.display_norms = display_norms
        self.E_s = nn.Embedding(self.entity_count, self.embedding_dim)
        self.E_o = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R = nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_inv = nn.Embedding(self.relation_count, self.embedding_dim)
        #'''
        nn.init.normal_(self.E_s.weight.data, 0, 0.05)
        nn.init.normal_(self.E_o.weight.data, 0, 0.05)
        nn.init.normal_(self.R.weight.data, 0, 0.05)

        '''
        sqrt_size = 6.0/math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.E_s.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.E_o.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.R.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.R_inv.weight.data, -sqrt_size, sqrt_size)
        #'''

        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        self.batch_norm = batch_norm
        if batch_norm:
            E_s_bn = nn.BatchNorm1d(self.embedding_dim)
            E_o_bn = nn.BatchNorm1d(self.embedding_dim)
            R_bn = nn.BatchNorm1d(self.embedding_dim)

        if flag_add_reverse:
            print("CX: Added inverse relations!")
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count

        self.flag_add_reverse = flag_add_reverse
        self.flag_avg_scores  = flag_avg_scores

    def forward(self, s, r, o, flag_debug=0):
        """
        This is the scoring function \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the scoring function
        """
        s_e_h = self.E_s(s) if s is not None else self.E_s.weight.unsqueeze(0)
        s_e_t = self.E_s(o) if o is not None else self.E_s.weight.unsqueeze(0)        

        r_e = self.R(r)
        
        '''
        o_e_h = self.E_o(o) if o is not None else self.E_o.weight.unsqueeze(0)
        o_e_t = self.E_o(s) if s is not None else self.E_o.weight.unsqueeze(0)
        '''
        o_e_t = self.E_o(o) if o is not None else self.E_o.weight.unsqueeze(0)
        o_e_h = self.E_o(s) if s is not None else self.E_o.weight.unsqueeze(0)


        r_e_inv = self.R_inv(r)
        #if s is None:
        #    pdb.set_trace()
        #if o is not None:
        #    if o.shape[0] == 10:
        #        pdb.set_trace()
   
             

        if o is None or o.shape[1] > 1: 
            tmp1 = s_e_h * r_e
            tmp1_inv = o_e_h * r_e_inv#o_e_t * r_e_inv
            if o is not None:
                #o_e_h = o_e_h[0].view(-1,self.embedding_dim).transpose(0,1)
                o_e_t = o_e_t[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_e_t#h

                s_e_t = s_e_t[0].view(-1,self.embedding_dim).transpose(0,1)
                result_inv = tmp1_inv @ s_e_t
            else:
                o_e_t = o_e_t.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_e_t

                s_e_t = s_e_t[0].view(-1, self.embedding_dim).transpose(0,1)
                result_inv = tmp1_inv @ s_e_t

            result = result.squeeze(1)
            result_inv = result_inv.squeeze(1)
        else:
            tmp1 = o_e_t * r_e
            tmp1_inv = s_e_t * r_e_inv
            if s is not None:
                s_e_h = s_e_h[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_e_h

                o_e_h = o_e_h[0].view(-1, self.embedding_dim).transpose(0,1)
                result_inv = tmp1_inv @ o_e_h
            else:
                s_e_h = s_e_h.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_e_h

                o_e_h = o_e_h[0].view(-1, self.embedding_dim).transpose(0,1)
                result_inv = tmp1_inv @ o_e_h
            result = result.squeeze(1)
            result_inv = result_inv.squeeze(1)

        #print("\nresult shape:{}, result_inv.shape:{}".format(result.shape, result_inv.shape))
        #xx=input()
        #return torch.clamp(result/2, -20, 20)

        #return torch.clamp(result_inv,-20,20)#
        #return torch.clamp(result,-20,20)#
        #return torch.clamp((result + result_inv)/2,-20,20)#.squeeze()
        return (result+result_inv)/2
        #return result


    def regularizer(self, s, r, o):
        """
        This is the regularization term \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the regularization term
        """
        s1 = self.E_s(s)
        r1 = self.R(r)
        o1 = self.E_o(o)

        o2 = self.E_s(o)
        r2 = self.R_inv(r)
        s2 = self.E_o(s)

        if self.reg==2:
            return (s1**2 + o1**2 + r1**2 + o2**2 + r2**2 + s2**2).sum()/2
        elif self.reg==22:
            #return ((torch.norm(self.E_s.weight, p=2) ** 2) + (torch.norm(self.E_o.weight, p=2) ** 2) + (torch.norm(self.R_inv.weight,p=2)**2) )
            #return ((torch.norm(self.E_s.weight, p=2) ** 2) + (torch.norm(self.E_o.weight, p=2) ** 2) + (torch.norm(self.R.weight, p=2) ** 2)) 
            return ((torch.norm(self.E_s.weight, p=2) ** 2) + (torch.norm(self.E_o.weight, p=2) ** 2) + (torch.norm(self.R.weight, p=2) ** 2) + (torch.norm(self.R_inv.weight,p=2)**2) ) 
            #(self.E_s.weight**2 + self.E_o.weight**2 + self.R.weight**2)/2
        elif self.reg == 1:
            return (s.abs()+r.abs()+o.abs()).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s**2),torch.sqrt(o**2),torch.sqrt(r**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg/s.shape[0]
        else:
            print("Unknown reg for distmult model")
            assert(False)

    def post_epoch(self):
        """
        Post epoch/batch processing stuff.
        :return: Any message that needs to be displayed after each batch
        """
        if(self.unit_reg):
            self.E_s.weight.data.div_(self.E_s.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_o.weight.data.div_(self.E_o.weight.data.norm(2, dim=-1, keepdim=True))
            self.R.weight.data.div_(self.R.weight.data.norm(2, dim=-1, keepdim=True))
        return ""



class SimplIE(nn.Module):
    """
    DistMult Model from Trullion et al 2014.\n
    Scoring function (s, r, o) = <s, r, o> # dot product
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False, reg=2, batch_norm=False, flag_add_reverse=0, flag_avg_scores=0):
        """
        The initializing function. These parameters are expected to be supplied from the command line when running the\n
        program from main.\n
        :param entity_count: The number of entities in the knowledge base/model
        :param relation_count: Number of relations in the knowledge base/model
        :param embedding_dim: The size of the embeddings of entities and relations
        :param unit_reg: Whether the ___entity___ embeddings should be unit regularized or not
        :param clamp_v: The value at which to clamp the scores. (necessary to avoid over/underflow with some losses
        :param display_norms: Whether to display the max and min entity and relation embedding norms with each update
        :param reg: The type of regularization (example-l1,l2)
        """
        super(SimplIE, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.reg = reg

        self.display_norms = display_norms
        self.E_s = nn.Embedding(self.entity_count, self.embedding_dim)
        self.E_o = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R = nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_inv = nn.Embedding(self.relation_count, self.embedding_dim)
        '''
        nn.init.normal_(self.E_s.weight.data, 0, 0.05)
        nn.init.normal_(self.E_o.weight.data, 0, 0.05)
        nn.init.normal_(self.R.weight.data, 0, 0.05)

        '''
        sqrt_size = 6.0/math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.E_s.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.E_o.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.R.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.R_inv.weight.data, -sqrt_size, sqrt_size)
        #'''

        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        self.batch_norm = batch_norm
        if batch_norm:
            E_s_bn = nn.BatchNorm1d(self.embedding_dim)
            E_o_bn = nn.BatchNorm1d(self.embedding_dim)
            R_bn = nn.BatchNorm1d(self.embedding_dim)

        if flag_add_reverse:
            print("CX: Added inverse relations!")
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count

        self.flag_add_reverse = flag_add_reverse
        self.flag_avg_scores  = flag_avg_scores

    def forward(self, s, r, o, flag_debug=0):
        """
        This is the scoring function \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the scoring function
        """
        s_e_h = self.E_s(s) if s is not None else self.E_s.weight.unsqueeze(0)
        s_e_t = self.E_s(o) if o is not None else self.E_s.weight.unsqueeze(0)        

        r_e = self.R(r)
        
        '''
        o_e_h = self.E_o(o) if o is not None else self.E_o.weight.unsqueeze(0)
        o_e_t = self.E_o(s) if s is not None else self.E_o.weight.unsqueeze(0)
        '''
        o_e_t = self.E_o(o) if o is not None else self.E_o.weight.unsqueeze(0)
        o_e_h = self.E_o(s) if s is not None else self.E_o.weight.unsqueeze(0)


        r_e_inv = self.R_inv(r)
        #if s is None:
        #    pdb.set_trace()
        #if o is not None:
        #    if o.shape[0] == 10:
        #        pdb.set_trace()
   
             

        if o is None or o.shape[1] > 1: 
            tmp1 = s_e_h * r_e
            tmp1_inv = o_e_h * r_e_inv#o_e_t * r_e_inv
            if o is not None:
                #o_e_h = o_e_h[0].view(-1,self.embedding_dim).transpose(0,1)
                o_e_t = o_e_t[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_e_t#h

                s_e_t = s_e_t[0].view(-1,self.embedding_dim).transpose(0,1)
                result_inv = tmp1_inv @ s_e_t
            else:
                o_e_t = o_e_t.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_e_t

                s_e_t = s_e_t[0].view(-1, self.embedding_dim).transpose(0,1)
                result_inv = tmp1_inv @ s_e_t

            result = result.squeeze(1)
            result_inv = result_inv.squeeze(1)
        else:
            tmp1 = o_e_t * r_e
            tmp1_inv = s_e_t * r_e_inv
            if s is not None:
                s_e_h = s_e_h[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_e_h

                o_e_h = o_e_h[0].view(-1, self.embedding_dim).transpose(0,1)
                result_inv = tmp1_inv @ o_e_h
            else:
                s_e_h = s_e_h.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_e_h

                o_e_h = o_e_h[0].view(-1, self.embedding_dim).transpose(0,1)
                result_inv = tmp1_inv @ o_e_h
            result = result.squeeze(1)
            result_inv = result_inv.squeeze(1)

        #print("\nresult shape:{}, result_inv.shape:{}".format(result.shape, result_inv.shape))
        #xx=input()
        #return torch.clamp(result/2, -20, 20)

        #return torch.clamp(result_inv,-20,20)#
        #return torch.clamp(result,-20,20)#
        return torch.clamp((result + result_inv)/2,-20,20)#.squeeze()

        #return result


    def regularizer(self, s, r, o):
        """
        This is the regularization term \n
        :param s: The entities corresponding to the subject position. Must be a torch long tensor of 2 dimensions batch * x
        :param r: The relations for the fact. Must be a torch long tensor of 2 dimensions batch * x
        :param o: The entities corresponding to the object position. Must be a torch long tensor of 2 dimensions batch * x
        :return: The computation graph corresponding to the forward pass of the regularization term
        """
        

        s1 = self.E_s(s)
        r1 = self.R(r)
        o1 = self.E_o(o)

        o2 = self.E_s(o)
        r2 = self.R_inv(r)
        s2 = self.E_o(s)

        if self.reg==2:
            return (s1**2 + o1**2 + r1**2 + o2**2 + r2**2 + s2**2).sum()/2
        elif self.reg==22:
            #return ((torch.norm(self.E_s.weight, p=2) ** 2) + (torch.norm(self.E_o.weight, p=2) ** 2) + (torch.norm(self.R_inv.weight,p=2)**2) )
            #return ((torch.norm(self.E_s.weight, p=2) ** 2) + (torch.norm(self.E_o.weight, p=2) ** 2) + (torch.norm(self.R.weight, p=2) ** 2)) 
            return ((torch.norm(self.E_s.weight, p=2) ** 2) + (torch.norm(self.E_o.weight, p=2) ** 2) + (torch.norm(self.R.weight, p=2) ** 2) + (torch.norm(self.R_inv.weight,p=2)**2) ) 
            #(self.E_s.weight**2 + self.E_o.weight**2 + self.R.weight**2)/2
        elif self.reg == 1:
            return (s.abs()+r.abs()+o.abs()).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s**2),torch.sqrt(o**2),torch.sqrt(r**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg/s.shape[0]
        else:
            print("Unknown reg for distmult model")
            assert(False)

    def post_epoch(self):
        """
        Post epoch/batch processing stuff.
        :return: Any message that needs to be displayed after each batch
        """
        if(self.unit_reg):
            self.E_s.weight.data.div_(self.E_s.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_o.weight.data.div_(self.E_o.weight.data.norm(2, dim=-1, keepdim=True))
            self.R.weight.data.div_(self.R.weight.data.norm(2, dim=-1, keepdim=True))
        return ""

class RotatE(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, gamma=0, clamp_v=None, reg=2,
                 batch_norm=False, unit_reg=False, has_cuda=True):

        super(RotatE, self).__init__()
        print("Gamma", gamma)
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count

        self.epsilon = 2.0
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.embedding_dim]), 
            requires_grad=False
        )

        self.E_im = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = nn.Embedding(self.relation_count, self.embedding_dim)

        nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        '''
        nn.init.normal_(self.E_re.weight.data, -self.embedding_range.item(),self.embedding_range.item())#0, 0.05)
        nn.init.normal_(self.E_im.weight.data, -self.embedding_range.item(),self.embedding_range.item())#0, 0.05)
        nn.init.normal_(self.R_re.weight.data, -self.embedding_range.item(),self.embedding_range.item())#0, 0.05)
        nn.init.normal_(self.R_im.weight.data, -self.embedding_range.item(),self.embedding_range.item())#0, 0.05)
        '''

        self.minimum_value = -self.embedding_dim * self.embedding_dim
        #self.clamp_v = clamp_v

        #self.unit_reg = unit_reg

        self.reg = reg
        print("Regularization value:", reg)
        print("embedding_dim value:", embedding_dim)


    #@profile
    def forward(self, s, r, o, flag_debug=0, using_sm=0):
        # if flag_debug==2:
        #     pdb.set_trace()

        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)

        #print(s_im.shape, o_im.shape, r_im.shape)

        #'''
        #def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        #re_head, im_head = torch.chunk(head, 2, dim=2)
        #re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_r_re = r_re/(self.embedding_range.item()/pi)
        phase_r_im = r_im/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_r_re)
        im_relation = torch.sin(phase_r_im)

        if s is None:#mode == 'head-batch':
            re_score = re_relation * o_re + im_relation * o_im
            im_score = re_relation * o_im - im_relation * o_re
            re_score = re_score - s_re
            im_score = im_score - s_im
        else:
            re_score = s_re * re_relation - s_im * im_relation
            im_score = s_re * im_relation + s_im * re_relation
            re_score = re_score - o_re
            im_score = im_score - o_im

        #pdb.set_trace()
        #print("ready!", re_score.shape, im_score.shape)
        #score = torch.stack([re_score, im_score], dim = 0)
        #score = score.norm(dim = 0)
        #score = torch.sqrt((re_score.to_sparse()**2 + im_score.to_sparse()**2).to_dense())
        score = torch.sqrt(re_score**2 + im_score**2)
        score = self.gamma.item() - score.sum(dim = 2)
        #'''
        #score = 0
        #if s is None:
        #    exit()
        return score


    def regularizer(self, s, r, o, reg_val=0):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        if reg_val:
            self.reg = reg_val
            # print("CX reg", reg_val)

        if self.reg == 2:
            return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s_re**2 + s_im**2),torch.sqrt(o_re**2+o_im**2),torch.sqrt(r_re**2+r_im**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg/s.shape[0]
        else:
            print("Unknown reg for complex model")
            assert(False)

'''
    def post_epoch(self):
        if(self.unit_reg):
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""
'''




class complex_reflexive(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, clamp_v=None, reg=2, batch_norm=False, flag_train_reflexive=1):
        super(complex_reflexive, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E_im = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = nn.Embedding(self.relation_count, self.embedding_dim)
        nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v

        self.reg = reg
        print("Regularization value:", reg)

        self.batch_norm = batch_norm

        if batch_norm:
            self.E_im_bn = nn.BatchNorm1d(self.embedding_dim)
            self.E_re_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_im_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_re_bn = nn.BatchNorm1d(self.embedding_dim)

        #reflexive
        self.flag_train_reflexive = flag_train_reflexive
        if self.flag_train_reflexive:
            print("Training reflexive values")
        self.eps = nn.Embedding(self.relation_count, 1)
        nn.init.constant_(self.eps.weight.data, -3.0)
        
        self.ent_idx= torch.from_numpy(np.arange(self.entity_count)).long().cuda() #list of all ent indices
        
        ##

    def forward(self, s, r, o, flag_debug=0):      

        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)
        
        if self.batch_norm:
            s_im = self.E_im_bn(s_im)
            s_re = self.E_im_bn(s_re)
            o_im = self.E_im_bn(o_im)
            o_re = self.E_im_bn(o_re)
            r_im = self.R_im_bn(r_im)
            r_re = self.R_im_bn(r_re)

        if self.clamp_v:
            s_im.data.clamp_(-self.clamp_v, self.clamp_v)
            s_re.data.clamp_(-self.clamp_v, self.clamp_v)
            r_im.data.clamp_(-self.clamp_v, self.clamp_v)
            r_re.data.clamp_(-self.clamp_v, self.clamp_v)
            o_im.data.clamp_(-self.clamp_v, self.clamp_v)
            o_re.data.clamp_(-self.clamp_v, self.clamp_v)
        

        eq=[]
        if s is None:
            eq=(o==self.ent_idx).type(torch.cuda.FloatTensor)  #s is num_objects x num_ents (all ents used as negative samples) 
        elif o is None:
            eq=(s==self.ent_idx).type(torch.cuda.FloatTensor)            
        else:
            eq=(s==o).type(torch.cuda.FloatTensor)


        ##all ent as neg samples
        if o is None or o.shape[1] > 1:
            tmp1 = (s_im*r_re+s_re*r_im); tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = (s_re*r_re-s_im*r_im); tmp2 = tmp2.view(-1,self.embedding_dim)

            if o is not  None:#o.shape[1] > 1:
                o_re = o_re[0].view(-1,self.embedding_dim).transpose(0,1)
                o_im = o_im[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_im + tmp2 @o_re
            else:
                o_re = o_re.view(-1,self.embedding_dim).transpose(0,1)
                o_im = o_im.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_im + tmp2 @o_re 
        else:
            tmp1 = o_im*r_re-o_re*r_im; tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = o_im*r_im+o_re*r_re; tmp2 = tmp2.view(-1,self.embedding_dim)

            if s is not None:#s.shape[1] > 1:
                s_im = s_im[0].view(-1,self.embedding_dim).transpose(0,1)
                s_re = s_re[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_im + tmp2 @ s_re
            else:
                s_im = s_im.view(-1,self.embedding_dim).transpose(0,1)
                s_re = s_re.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_im + tmp2 @ s_re


        eps_wts=torch.unsqueeze(self.eps(r).squeeze(),1) 

        score_new = eps_wts * (result * eq) + (result * (eq!=1).type(torch.cuda.FloatTensor) )


        if flag_debug:
            print("Debug: ", torch.mean(eps_wts), torch.std(eps_wts))
            a = torch.sum(eps_wts * eq.type(torch.cuda.FloatTensor))
            b = torch.sum(eq.type(torch.cuda.FloatTensor))
            print("Debug: ", a, b, a/b)
            a = torch.sum(eps_wts * (eq!=1).type(torch.cuda.FloatTensor))
            b = torch.sum((eq!=1).type(torch.cuda.FloatTensor))
            print("Debug: ", a, b, a/b)

        return score_new

    def regularizer(self, s, r, o, reg_val=0):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        if reg_val:
            self.reg = reg_val
        if self.reg == 2:
            return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s_re**2 + s_im**2),torch.sqrt(o_re**2+o_im**2),torch.sqrt(r_re**2+r_im**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg/s.shape[0]
        else:
            print("Unknown reg for complex model")
            assert(False)


    def post_epoch(self):
        return ""

class complex_lx_old(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, clamp_v=None, reg=3, batch_norm=False, flag_add_reverse=1):
        super(complex_lx, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E_im = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = nn.Embedding(self.relation_count, self.embedding_dim)
        nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v

        self.reg = reg
        print("Regularization value:", reg)

        self.batch_norm = batch_norm

        print("batch_norm", batch_norm, self.batch_norm)

        if batch_norm:
            self.E_im_bn = nn.BatchNorm1d(self.embedding_dim)
            self.E_re_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_im_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_re_bn = nn.BatchNorm1d(self.embedding_dim)

        if flag_add_reverse:
            print("CX: Added inverse relations!")
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count
        self.flag_add_reverse=flag_add_reverse



    def forward(self, s, r, o, flag_debug=0):

        ##
        if self.flag_add_reverse:
            total_rel = torch.tensor(self.relation_count).cuda()
            inv_or_not = r >= total_rel; #inv_or_not = inv_or_not.type(torch.LongTensor)
            r = r - inv_or_not.type(torch.cuda.LongTensor) * total_rel
        ##

        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)

        if self.batch_norm:
            s_im = self.E_im_bn(s_im.squeeze())
            s_re = self.E_im_bn(s_re.squeeze())
            o_im = self.E_im_bn(o_im.squeeze())
            o_re = self.E_im_bn(o_re.squeeze())
            r_im = self.R_im_bn(r_im.squeeze())
            r_re = self.R_im_bn(r_re.squeeze())
            dim_squeeze = 1
            r_im = r_im.unsqueeze(dim_squeeze)
            r_re = r_re.unsqueeze(dim_squeeze)
            if s is None:
                dim_squeeze = 0
            s_im = s_im.unsqueeze(dim_squeeze)
            s_re = s_re.unsqueeze(dim_squeeze)
            dim_squeeze = 1
            if o is None:
                dim_squeeze = 0
            o_im = o_im.unsqueeze(dim_squeeze)
            o_re = o_re.unsqueeze(dim_squeeze)

        if self.clamp_v:
            s_im.data.clamp_(-self.clamp_v, self.clamp_v)
            s_re.data.clamp_(-self.clamp_v, self.clamp_v)
            r_im.data.clamp_(-self.clamp_v, self.clamp_v)
            r_re.data.clamp_(-self.clamp_v, self.clamp_v)
            o_im.data.clamp_(-self.clamp_v, self.clamp_v)
            o_re.data.clamp_(-self.clamp_v, self.clamp_v)
        ##all ent as neg samples
        if o is None or o.shape[1] > 1:
            tmp1 = (s_im*r_re+s_re*r_im); tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = (s_re*r_re-s_im*r_im); tmp2 = tmp2.view(-1,self.embedding_dim)

            if o is not  None:#o.shape[1] > 1:
                o_re = o_re[0].view(-1,self.embedding_dim).transpose(0,1)
                o_im = o_im[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_im + tmp2 @o_re
            else:
                o_re = o_re.view(-1,self.embedding_dim).transpose(0,1)
                o_im = o_im.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_im + tmp2 @o_re 
        else:
            tmp1 = o_im*r_re-o_re*r_im; tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = o_im*r_im+o_re*r_re; tmp2 = tmp2.view(-1,self.embedding_dim)

            if s is not None:#s.shape[1] > 1:
                s_im = s_im[0].view(-1,self.embedding_dim).transpose(0,1)
                s_re = s_re[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_im + tmp2 @ s_re
            else:
                s_im = s_im.view(-1,self.embedding_dim).transpose(0,1)
                s_re = s_re.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_im + tmp2 @ s_re

        return result
        #'''

    def regularizer(self, s, r, o, reg_val=0):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        if reg_val:
            self.reg = reg_val
        if self.reg == 2:
            return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s_re**2 + s_im**2),torch.sqrt(o_re**2+o_im**2),torch.sqrt(r_re**2+r_im**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg/s.shape[0]
        else:
            print("Unknown reg for complex model")
            assert(False)


    def post_epoch(self):
        return ""


class complex_lx(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, clamp_v=None, reg=2,
                 batch_norm=False, unit_reg=False, has_cuda=True, flag_add_reverse=1, flag_avg_scores=0):

        super(complex_lx, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E_im = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = nn.Embedding(self.relation_count, self.embedding_dim)
        nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim * self.embedding_dim
        self.clamp_v = clamp_v

        self.unit_reg = unit_reg

        self.reg = reg
        print("Regularization value:", reg)
        print("embedding_dim value:", embedding_dim)
        self.batch_norm = batch_norm

        print("batch_norm", batch_norm, self.batch_norm)

        if batch_norm:
            self.E_im_bn = nn.BatchNorm1d(self.embedding_dim)
            self.E_re_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_im_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_re_bn = nn.BatchNorm1d(self.embedding_dim)

        if flag_add_reverse:
            print("CX: Added inverse relations!")
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count


        self.flag_add_reverse = flag_add_reverse
        self.flag_avg_scores  = flag_avg_scores


    def forward(self, s, r, o, flag_debug=0):

        ##
        if 0:#self.flag_add_reverse:
            total_rel = torch.tensor(self.relation_count).cuda()
            inv_or_not = r >= total_rel; #inv_or_not = inv_or_not.type(torch.LongTensor)
            r = r - inv_or_not.type(torch.cuda.LongTensor) * total_rel
        ##
        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)

        if self.batch_norm:
            s_im = self.E_im_bn(s_im.squeeze())
            s_re = self.E_im_bn(s_re.squeeze())
            o_im = self.E_im_bn(o_im.squeeze())
            o_re = self.E_im_bn(o_re.squeeze())
            r_im = self.R_im_bn(r_im.squeeze())
            r_re = self.R_im_bn(r_re.squeeze())
            dim_squeeze = 1
            r_im = r_im.unsqueeze(dim_squeeze)
            r_re = r_re.unsqueeze(dim_squeeze)
            if s is None:
                dim_squeeze = 0
            s_im = s_im.unsqueeze(dim_squeeze)
            s_re = s_re.unsqueeze(dim_squeeze)
            dim_squeeze = 1
            if o is None:
                dim_squeeze = 0
            o_im = o_im.unsqueeze(dim_squeeze)
            o_re = o_re.unsqueeze(dim_squeeze)

        if self.clamp_v:
            s_im.data.clamp_(-self.clamp_v, self.clamp_v)
            s_re.data.clamp_(-self.clamp_v, self.clamp_v)
            r_im.data.clamp_(-self.clamp_v, self.clamp_v)
            r_re.data.clamp_(-self.clamp_v, self.clamp_v)
            o_im.data.clamp_(-self.clamp_v, self.clamp_v)
            o_re.data.clamp_(-self.clamp_v, self.clamp_v)
        
        result = None
        if o is None or o.shape[1] > 1:
            tmp1 = (s_im * r_re + s_re * r_im);
            tmp1 = tmp1.view(-1, self.embedding_dim)
            tmp2 = (s_re * r_re - s_im * r_im);
            tmp2 = tmp2.view(-1, self.embedding_dim)

            if o is not None:  # o.shape[1] > 1:
                result = (tmp1 * o_im + tmp2 * o_re).sum(dim=-1)
            else:  # all ent as neg samples
                o_re = o_re.view(-1, self.embedding_dim).transpose(0, 1)
                o_im = o_im.view(-1, self.embedding_dim).transpose(0, 1)
                result = tmp1 @ o_im + tmp2 @ o_re
        elif s is None or s.shape[1] > 1:
            tmp1 = o_im * r_re - o_re * r_im;
            tmp1 = tmp1.view(-1, self.embedding_dim)
            tmp2 = o_im * r_im + o_re * r_re;
            tmp2 = tmp2.view(-1, self.embedding_dim)

            if s is not None: 
                result = (tmp1 * s_im + tmp2 * s_re).sum(dim=-1)
            else:
                s_im = s_im.view(-1, self.embedding_dim).transpose(0, 1)
                s_re = s_re.view(-1, self.embedding_dim).transpose(0, 1)
                result = tmp1 @ s_im + tmp2 @ s_re

        elif r is None:  # relation prediction
            tmp1 = o_im * s_re - o_re * s_im;
            tmp1 = tmp1.view(-1, self.embedding_dim)
            tmp2 = o_im * s_im + o_re * s_re;
            tmp2 = tmp2.view(-1, self.embedding_dim)

            r_im = r_im.view(-1, self.embedding_dim).transpose(0, 1)
            r_re = r_re.view(-1, self.embedding_dim).transpose(0, 1)

            result = tmp1 @ r_im + tmp2 @ r_re
        
        if (result is None):
            result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
            result.sum(dim=-1)

        return result

    def regularizer(self, s, r, o, reg_val=0):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        if reg_val:
            self.reg = reg_val
            # print("CX reg", reg_val)

        if self.reg == 2:
            return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s_re**2 + s_im**2),torch.sqrt(o_re**2+o_im**2),torch.sqrt(r_re**2+r_im**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg/s.shape[0]
        else:
            print("Unknown reg for complex model")
            assert(False)


    def post_epoch(self):
        if(self.unit_reg):
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""




class complex_tim(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, clamp_v=None, reg=2, batch_norm=False, unit_reg=False, input_dropout=0.2):
        super(complex_tim, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E_im = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = nn.Embedding(self.relation_count, self.embedding_dim)

        nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v

        self.unit_reg = unit_reg

        self.reg = reg
        print("Regularization value:", reg)

        self.batch_norm = batch_norm

        print("batch_norm", batch_norm, self.batch_norm)

        if batch_norm:
            self.E_im_bn = nn.BatchNorm1d(self.embedding_dim)
            self.E_re_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_im_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_re_bn = nn.BatchNorm1d(self.embedding_dim)

    def forward(self, s, r, o, flag_debug=0):
        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)

        if self.batch_norm:
            s_im = self.E_im_bn(s_im.squeeze())
            s_re = self.E_im_bn(s_re.squeeze())
            o_im = self.E_im_bn(o_im.squeeze())
            o_re = self.E_im_bn(o_re.squeeze())
            r_im = self.R_im_bn(r_im.squeeze())
            r_re = self.R_im_bn(r_re.squeeze())
            dim_squeeze = 1
            r_im = r_im.unsqueeze(dim_squeeze)
            r_re = r_re.unsqueeze(dim_squeeze)
            if s is None:
                dim_squeeze = 0
            s_im = s_im.unsqueeze(dim_squeeze)
            s_re = s_re.unsqueeze(dim_squeeze)
            dim_squeeze = 1
            if o is None:
                dim_squeeze = 0
            o_im = o_im.unsqueeze(dim_squeeze)
            o_re = o_re.unsqueeze(dim_squeeze)


        if self.clamp_v:
            s_im.data.clamp_(-self.clamp_v, self.clamp_v)
            s_re.data.clamp_(-self.clamp_v, self.clamp_v)
            r_im.data.clamp_(-self.clamp_v, self.clamp_v)
            r_re.data.clamp_(-self.clamp_v, self.clamp_v)
            o_im.data.clamp_(-self.clamp_v, self.clamp_v)
            o_re.data.clamp_(-self.clamp_v, self.clamp_v)

        #'''
        ##all ent as neg samples
        if o is None or o.shape[1] > 1:
            tmp1 = (s_im*r_re+s_re*r_im); tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = (s_re*r_re-s_im*r_im); tmp2 = tmp2.view(-1,self.embedding_dim)

            if o is not  None:#o.shape[1] > 1:
                o_re = o_re[0].view(-1,self.embedding_dim).transpose(0,1)
                o_im = o_im[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_im + tmp2 @o_re
            else:
                o_re = o_re.view(-1,self.embedding_dim).transpose(0,1)
                o_im = o_im.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_im + tmp2 @o_re 
        else:
            tmp1 = o_im*r_re-o_re*r_im; tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = o_im*r_im+o_re*r_re; tmp2 = tmp2.view(-1,self.embedding_dim)

            if s is not None:#s.shape[1] > 1:
                s_im = s_im[0].view(-1,self.embedding_dim).transpose(0,1)
                s_re = s_re[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_im + tmp2 @ s_re
            else:
                s_im = s_im.view(-1,self.embedding_dim).transpose(0,1)
                s_re = s_re.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_im + tmp2 @ s_re

        return torch.sigmoid(result)
        #'''

    def regularizer(self, s, r, o, reg_val=0):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        if reg_val:
            self.reg = reg_val
            # print("CX reg", reg_val)

        if self.reg == 2:
            return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s_re**2 + s_im**2),torch.sqrt(o_re**2+o_im**2),torch.sqrt(r_re**2+r_im**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg/s.shape[0]
        else:
            print("Unknown reg for complex model")
            assert(False)


    def post_epoch(self):
        if(self.unit_reg):
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""


class complex_fast(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, clamp_v=None, reg=2,
                 batch_norm=False, unit_reg=False, has_cuda=True):
        super(complex_fast, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E_im = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = nn.Embedding(self.relation_count, self.embedding_dim)
        nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim * self.embedding_dim
        self.clamp_v = clamp_v
        self.unit_reg = unit_reg
        self.reg = reg

        print("Regularization value:", reg)
        print("embedding_dim value:", embedding_dim)
        self.batch_norm = batch_norm
        print("batch_norm", batch_norm, self.batch_norm)

    #@profile
    def forward(self, s, r, o, flag_debug=0, using_sm=0):
        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)

        if self.clamp_v:
            s_im.data.clamp_(-self.clamp_v, self.clamp_v)
            s_re.data.clamp_(-self.clamp_v, self.clamp_v)
            r_im.data.clamp_(-self.clamp_v, self.clamp_v)
            r_re.data.clamp_(-self.clamp_v, self.clamp_v)
            o_im.data.clamp_(-self.clamp_v, self.clamp_v)
            o_re.data.clamp_(-self.clamp_v, self.clamp_v)

        # if flag_debug==2:
        #     pdb.set_trace()
        
        all_e_im = self.E_im.weight.unsqueeze(0).view(-1, self.embedding_dim).transpose(0, 1)
        all_e_re = self.E_re.weight.unsqueeze(0).view(-1, self.embedding_dim).transpose(0, 1)

        result = None
        if o is None or o.shape[1] > 1:
            tmp1 = (s_im * r_re + s_re * r_im);
            tmp2 = (s_re * r_re - s_im * r_im);
           
            result_all = tmp1 @ all_e_im + tmp2 @ all_e_re 
 
            if o is not None or using_sm: 
                result = result_all.squeeze().gather(1,o)
                #pdb.set_trace()
                #result = result_all.gather(1,o.view(-1,1))
                #result = result_all[o]
            else:  # all ent as neg samples
                result = result_all  

        elif s is None or s.shape[1] > 1:
            tmp1 = o_im * r_re - o_re * r_im;
            tmp2 = o_im * r_im + o_re * r_re;

            result_all = tmp1 @ all_e_im + tmp2 @ all_e_re

            if s is not None or using_sm: 
                result = result_all.squeeze().gather(1,s)
                #result = (tmp1 * s_im + tmp2 * s_re).sum(dim=-1)
            else:
                result = result_all
                #s_im = s_im.view(-1, self.embedding_dim).transpose(0, 1)
                #s_re = s_re.view(-1, self.embedding_dim).transpose(0, 1)
                #result = tmp1 @ s_im + tmp2 @ s_re

        
        if (result is None):
            result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
            result = result.sum(dim=-1)

        result = result.view(r.shape[0],-1)
        return result

    #@profile
    def regularizer(self, s, r, o, reg_val=0):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        if reg_val:
            self.reg = reg_val
            # print("CX reg", reg_val)

        if self.reg == 2:
            return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s_re**2 + s_im**2),torch.sqrt(o_re**2+o_im**2),torch.sqrt(r_re**2+r_im**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)

            #exit()

            return reg/s.shape[0]
        
        else:
            print("Unknown reg for complex model")
            assert(False)


    def post_epoch(self):
        if(self.unit_reg):
            self.E_im.weight.data.div_(torch.sum(self.E_im.weight.data.abs(), dim=-1))
            self.E_re.weight.data.div_(torch.sum(self.E_re.weight.data.abs(), dim=-1))#self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(torch.sum(self.R_im.weight.data.abs(), dim=-1))#self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(torch.sum(self.R_re.weight.data.abs(), dim=-1))#self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
            '''
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
            '''
        return ""


class complex(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, clamp_v=None, reg=2,
                 batch_norm=False, unit_reg=False, has_cuda=True):
        super(complex, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E_im = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = nn.Embedding(self.relation_count, self.embedding_dim)
        nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim * self.embedding_dim
        self.clamp_v = clamp_v
        self.unit_reg = unit_reg
        self.reg = reg

        print("Regularization value:", reg)
        print("embedding_dim value:", embedding_dim)
        self.batch_norm = batch_norm
        print("batch_norm", batch_norm, self.batch_norm)

        if batch_norm:
            self.E_im_bn = nn.BatchNorm1d(self.embedding_dim)
            self.E_re_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_im_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_re_bn = nn.BatchNorm1d(self.embedding_dim)

    #@profile
    def forward(self, s, r, o, flag_debug=0, using_sm=0):
        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r) if r is not None else self.R_im.weight.unsqueeze(0)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r) if r is not None else self.R_re.weight.unsqueeze(0)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)

        if self.batch_norm:
            s_im = self.E_im_bn(s_im.squeeze())
            s_re = self.E_im_bn(s_re.squeeze())
            o_im = self.E_im_bn(o_im.squeeze())
            o_re = self.E_im_bn(o_re.squeeze())
            r_im = self.R_im_bn(r_im.squeeze())
            r_re = self.R_im_bn(r_re.squeeze())
            dim_squeeze = 1
            r_im = r_im.unsqueeze(dim_squeeze)
            r_re = r_re.unsqueeze(dim_squeeze)
            if s is None:
                dim_squeeze = 0
            s_im = s_im.unsqueeze(dim_squeeze)
            s_re = s_re.unsqueeze(dim_squeeze)
            dim_squeeze = 1
            if o is None:
                dim_squeeze = 0
            o_im = o_im.unsqueeze(dim_squeeze)
            o_re = o_re.unsqueeze(dim_squeeze)
            
            if r is None:
                dim_squeeze = 0
            r_im = r_im.unsqueeze(dim_squeeze)
            r_re = r_re.unsqueeze(dim_squeeze)

        if self.clamp_v:
            s_im.data.clamp_(-self.clamp_v, self.clamp_v)
            s_re.data.clamp_(-self.clamp_v, self.clamp_v)
            r_im.data.clamp_(-self.clamp_v, self.clamp_v)
            r_re.data.clamp_(-self.clamp_v, self.clamp_v)
            o_im.data.clamp_(-self.clamp_v, self.clamp_v)
            o_re.data.clamp_(-self.clamp_v, self.clamp_v)

        # if flag_debug==2:
        #     pdb.set_trace()
        
        result = None
        if o is None or o.shape[1] > 1:
            tmp1 = (s_im * r_re + s_re * r_im);
            if 0:#not using_sm:
                tmp1 = tmp1.view(-1, self.embedding_dim)
            tmp2 = (s_re * r_re - s_im * r_im);
            if 0:#not using_sm:
                tmp2 = tmp2.view(-1, self.embedding_dim)
            
            if o is not None or using_sm:  # o.shape[1] > 1:
                #pdb.set_trace()
                result = (tmp1 * o_im + tmp2 * o_re).sum(dim=-1)
            else:  # all ent as neg samples
                o_re = o_re.view(-1, self.embedding_dim).transpose(0, 1)
                o_im = o_im.view(-1, self.embedding_dim).transpose(0, 1)
                result = tmp1 @ o_im + tmp2 @ o_re
        elif s is None or s.shape[1] > 1:
            tmp1 = o_im * r_re - o_re * r_im;
            if 0:#not using_sm:
                tmp1 = tmp1.view(-1, self.embedding_dim)
            tmp2 = o_im * r_im + o_re * r_re;
            if 0:#not using_sm:
                tmp2 = tmp2.view(-1, self.embedding_dim)

            if s is not None or using_sm: 
                result = (tmp1 * s_im + tmp2 * s_re).sum(dim=-1)
            else:
                s_im = s_im.view(-1, self.embedding_dim).transpose(0, 1)
                s_re = s_re.view(-1, self.embedding_dim).transpose(0, 1)
                result = tmp1 @ s_im + tmp2 @ s_re

        elif r is None:  # relation prediction
            tmp1 = o_im * s_re - o_re * s_im;
            tmp1 = tmp1.view(-1, self.embedding_dim)
            tmp2 = o_im * s_im + o_re * s_re;
            tmp2 = tmp2.view(-1, self.embedding_dim)

            r_im = r_im.view(-1, self.embedding_dim).transpose(0, 1)
            r_re = r_re.view(-1, self.embedding_dim).transpose(0, 1)

            result = tmp1 @ r_im + tmp2 @ r_re
        
        if (result is None):
            result = (s_re * o_re + s_im * o_im) * r_re + (s_re * o_im - s_im * o_re) * r_im
            result = result.sum(dim=-1)
        result = result.view(result.shape[0],-1)
        return result

    def score_subsampled_entities(self, s, r, o,
                                  head_candidates, tail_candidates):
        """Score the positives and only the shared candidates used by loss.

        ``crossentropy_loss_AllNeg_subsample`` discards every other column of
        the full entity-score matrix.  Computing these columns directly is the
        same ComplEx expression, with dramatically lower memory and FLOPs.
        This optimized path is intentionally limited to the release's formal
        no-batchnorm/no-clamp configuration.
        """
        if self.batch_norm or self.clamp_v:
            raise RuntimeError("subsampled score path requires no BN/clamp")
        s_im = self.E_im(s).squeeze(1)
        s_re = self.E_re(s).squeeze(1)
        r_im = self.R_im(r).squeeze(1)
        r_re = self.R_re(r).squeeze(1)
        o_im = self.E_im(o).squeeze(1)
        o_re = self.E_re(o).squeeze(1)

        tail_im = s_im * r_re + s_re * r_im
        tail_re = s_re * r_re - s_im * r_im
        tail_positive = (tail_im * o_im + tail_re * o_re).sum(dim=-1)
        tail_negative = (
            tail_im @ self.E_im(tail_candidates).t()
            + tail_re @ self.E_re(tail_candidates).t())

        head_im = o_im * r_re - o_re * r_im
        head_re = o_im * r_im + o_re * r_re
        head_positive = (head_im * s_im + head_re * s_re).sum(dim=-1)
        head_negative = (
            head_im @ self.E_im(head_candidates).t()
            + head_re @ self.E_re(head_candidates).t())
        return head_positive, head_negative, tail_positive, tail_negative

    #@profile
    def regularizer(self, s, r, o, reg_val=0):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        if reg_val:
            self.reg = reg_val
            # print("CX reg", reg_val)

        if self.reg == 2:
            return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s_re**2 + s_im**2),torch.sqrt(o_re**2+o_im**2),torch.sqrt(r_re**2+r_im**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)

            #exit()

            return reg/s.shape[0]
        
        else:
            print("Unknown reg for complex model")
            assert(False)


    def post_epoch(self):
        if(self.unit_reg):
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""



class complex_old(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, clamp_v=None, reg=2, batch_norm=False, unit_reg=False):
        super(complex, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E_im = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_im = nn.Embedding(self.relation_count, self.embedding_dim)
        self.E_re = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_re = nn.Embedding(self.relation_count, self.embedding_dim)
        nn.init.normal_(self.E_re.weight.data, 0, 0.05)
        nn.init.normal_(self.E_im.weight.data, 0, 0.05)
        nn.init.normal_(self.R_re.weight.data, 0, 0.05)
        nn.init.normal_(self.R_im.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v

        self.unit_reg = unit_reg

        self.reg = reg
        print("Regularization value:", reg)

        self.batch_norm = batch_norm

        print("batch_norm", batch_norm, self.batch_norm)

        if batch_norm:
            self.E_im_bn = nn.BatchNorm1d(self.embedding_dim)
            self.E_re_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_im_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_re_bn = nn.BatchNorm1d(self.embedding_dim)

    def forward(self, s, r, o, flag_debug=0):
        s_im = self.E_im(s) if s is not None else self.E_im.weight.unsqueeze(0)
        r_im = self.R_im(r)
        o_im = self.E_im(o) if o is not None else self.E_im.weight.unsqueeze(0)
        s_re = self.E_re(s) if s is not None else self.E_re.weight.unsqueeze(0)
        r_re = self.R_re(r)
        o_re = self.E_re(o) if o is not None else self.E_re.weight.unsqueeze(0)

        if self.batch_norm:
            s_im = self.E_im_bn(s_im.squeeze())
            s_re = self.E_im_bn(s_re.squeeze())
            o_im = self.E_im_bn(o_im.squeeze())
            o_re = self.E_im_bn(o_re.squeeze())
            r_im = self.R_im_bn(r_im.squeeze())
            r_re = self.R_im_bn(r_re.squeeze())
            dim_squeeze = 1
            r_im = r_im.unsqueeze(dim_squeeze)
            r_re = r_re.unsqueeze(dim_squeeze)
            if s is None:
                dim_squeeze = 0
            s_im = s_im.unsqueeze(dim_squeeze)
            s_re = s_re.unsqueeze(dim_squeeze)
            dim_squeeze = 1
            if o is None:
                dim_squeeze = 0
            o_im = o_im.unsqueeze(dim_squeeze)
            o_re = o_re.unsqueeze(dim_squeeze)

        if self.clamp_v:
            s_im.data.clamp_(-self.clamp_v, self.clamp_v)
            s_re.data.clamp_(-self.clamp_v, self.clamp_v)
            r_im.data.clamp_(-self.clamp_v, self.clamp_v)
            r_re.data.clamp_(-self.clamp_v, self.clamp_v)
            o_im.data.clamp_(-self.clamp_v, self.clamp_v)
            o_re.data.clamp_(-self.clamp_v, self.clamp_v)
        '''
        result = (s_re*o_re+s_im*o_im)*r_re + (s_re*o_im-s_im*o_re)*r_im
        return result.sum(dim=-1)
        '''

        '''
        if o is None or o.shape[1] > 1:
            tmp1 = (s_im*r_re+s_re*r_im)
            tmp2 = (s_re*r_re-s_im*r_im)
            result = tmp1 * o_im + tmp2 * o_re
        else:
            tmp1 = o_im*r_re-o_re*r_im;
            tmp2 = o_im*r_im+o_re*r_re;
            result = tmp1 * s_im + tmp2 * s_re

        return result.sum(dim=-1)
        '''
        '''
        ## Only using top neg-sample
        if o is None or o.shape[1] > 1:
            tmp1 = (s_im*r_re+s_re*r_im); tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = (s_re*r_re-s_im*r_im); tmp2 = tmp2.view(-1,self.embedding_dim)

            if o is not  None:#o.shape[1] > 1:
                o_re = o_re[0].view(-1,self.embedding_dim).transpose(0,1)
                o_im = o_im[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_im + tmp2 @o_re
            else:
                o_re = o_re.view(-1,self.embedding_dim).transpose(0,1)
                o_im = o_im.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_im + tmp2 @o_re 
        else:
            tmp1 = o_im*r_re-o_re*r_im; tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = o_im*r_im+o_re*r_re; tmp2 = tmp2.view(-1,self.embedding_dim)

            if s is not None:#s.shape[1] > 1:
                s_im = s_im[0].view(-1,self.embedding_dim).transpose(0,1)
                s_re = s_re[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_im + tmp2 @ s_re
            else:
                s_im = s_im.view(-1,self.embedding_dim).transpose(0,1)
                s_re = s_re.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_im + tmp2 @ s_re

        return result
        ##
        '''

        #'''
        ##all ent as neg samples
        if o is None or o.shape[1] > 1:
            tmp1 = (s_im*r_re+s_re*r_im); tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = (s_re*r_re-s_im*r_im); tmp2 = tmp2.view(-1,self.embedding_dim)

            if o is not  None:#o.shape[1] > 1:
                result = (tmp1 * o_im + tmp2 * o_re).sum(dim=-1)
                #o_re = o_re[0].view(-1,self.embedding_dim).transpose(0,1)
                #o_im = o_im[0].view(-1,self.embedding_dim).transpose(0,1)
                #result = tmp1 @ o_im + tmp2 @o_re
            else:
                o_re = o_re.view(-1,self.embedding_dim).transpose(0,1)
                o_im = o_im.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ o_im + tmp2 @o_re 
        else:
            tmp1 = o_im*r_re-o_re*r_im; tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = o_im*r_im+o_re*r_re; tmp2 = tmp2.view(-1,self.embedding_dim)

            if s is not None:#s.shape[1] > 1:
                s_im = s_im[0].view(-1,self.embedding_dim).transpose(0,1)
                s_re = s_re[0].view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_im + tmp2 @ s_re
            else:
                s_im = s_im.view(-1,self.embedding_dim).transpose(0,1)
                s_re = s_re.view(-1,self.embedding_dim).transpose(0,1)
                result = tmp1 @ s_im + tmp2 @ s_re

        return result
        #'''

        '''
        ##Old won't work
        if o is None or o.shape[1] > 1:
            tmp1 = (s_im*r_re+s_re*r_im); tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = (s_re*r_re-s_im*r_im); tmp2 = tmp2.view(-1,self.embedding_dim)
            o_re = o_re.view(-1,self.embedding_dim).transpose(0,1)
            o_im = o_im.view(-1,self.embedding_dim).transpose(0,1)
            result = tmp1 @ o_im + tmp2 @o_re 
        else:
            tmp1 = o_im*r_re-o_re*r_im; tmp1 = tmp1.view(-1,self.embedding_dim)
            tmp2 = o_im*r_im+o_re*r_re; tmp2 = tmp2.view(-1,self.embedding_dim)
            s_im = s_im.view(-1,self.embedding_dim).transpose(0,1)
            s_re = s_re.view(-1,self.embedding_dim).transpose(0,1)
            result = tmp1 @ s_im + tmp2 @ s_re

        if 0:#flag_debug:
            print("@Debug", "result, mean, std",torch.mean(result),torch.std(result))
        return result
        '''
    def regularizer(self, s, r, o, reg_val=0):
        s_im = self.E_im(s)
        r_im = self.R_im(r)
        o_im = self.E_im(o)
        s_re = self.E_re(s)
        r_re = self.R_re(r)
        o_re = self.E_re(o)
        if reg_val:
            self.reg = reg_val
            # print("CX reg", reg_val)

        if self.reg == 2:
            return (s_re**2+o_re**2+r_re**2+s_im**2+r_im**2+o_im**2).sum()
        elif self.reg == 3:
            factor = [torch.sqrt(s_re**2 + s_im**2),torch.sqrt(o_re**2+o_im**2),torch.sqrt(r_re**2+r_im**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            return reg/s.shape[0]
        else:
            print("Unknown reg for complex model")
            assert(False)


    def post_epoch(self):
        if(self.unit_reg):
            self.E_im.weight.data.div_(self.E_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.E_re.weight.data.div_(self.E_re.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_im.weight.data.div_(self.R_im.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_re.weight.data.div_(self.R_re.weight.data.norm(2, dim=-1, keepdim=True))
        return ""


class adder_model(nn.Module):
    def __init__(self, entity_count, relation_count, model1_name, model1_arguments, model2_name, model2_arguments):
        super(adder_model, self).__init__()

        model1 = globals()[model1_name]
        model1_arguments['entity_count'] = entity_count
        model1_arguments['relation_count'] = relation_count
        model2 = globals()[model2_name]
        model2_arguments['entity_count'] = entity_count
        model2_arguments['relation_count'] = relation_count

        self.model1 = model1(**model1_arguments)
        self.model2 = model2(**model2_arguments)
        self.minimum_value = self.model1.minimum_value + self.model2.minimum_value

    def forward(self, s, r, o):
        return self.model1(s, r, o) + self.model2(s, r, o)

    def post_epoch(self):
        return self.model1.post_epoch()+self.model2.post_epoch()

    def regularizer(self, s, r, o):
        return self.model1.regularizer(s, r, o) + self.model2.regularizer(s, r, o)

class typed_model(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, reg=2, batch_norm=False, psi_h=1.0, psi_t=1.0):
        super(typed_model, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        base_model_arguments['batch_norm'] = 0#batch_norm
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.psi_h = psi_h
        self.psi_t = psi_t

        self.E_t = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_ht = nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt = nn.Embedding(self.relation_count, self.embedding_dim)
        #'''
        nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        '''
        self.E_t.weight.data *= 1e-3
        self.R_ht.weight.data *= 1e-3
        self.R_tt.weight.data *= 1e-3
        '''
        self.minimum_value = 0.0


        print("typed model: batch_norm", batch_norm, batch_norm)

        self.reg = reg
        print("typed model reg", reg)
        self.batch_norm = batch_norm
        if batch_norm:
            self.E_t_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_ht_bn = nn.BatchNorm1d(self.embedding_dim)
            self.R_tt_bn = nn.BatchNorm1d(self.embedding_dim)

    def forward(self, s, r, o, flag_debug=0):
        base_forward = self.base_model(s, r, o)
        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)

        '''#old
        head_type_compatibility = (s_t*r_ht).sum(-1)
        tail_type_compatibility = (o_t*r_tt).sum(-1)
        '''

        if self.batch_norm:
            '''
            r_re = self.R_im_bn(r_re.squeeze())
            dim_squeeze = 1
            r_im = r_im.unsqueeze(dim_squeeze)
            r_re = r_re.unsqueeze(dim_squeeze)
            if s is None:
                dim_squeeze = 0
            s_im = s_im.unsqueeze(dim_squeeze)
            s_re = s_re.unsqueeze(dim_squeeze)
            dim_squeeze = 1
            if o is None:
                dim_squeeze = 0
            o_im = o_im.unsqueeze(dim_squeeze)
            o_re = o_re.unsqueeze(dim_squeeze)

            '''
            '''
            print("o_t", o_t.shape)
            print("s_t", s_t.shape)
            print("r_ht", r_ht.shape)
            '''
            o_t = self.E_t_bn(o_t.squeeze())
            s_t = self.E_t_bn(s_t.squeeze())
            r_ht= self.R_ht_bn(r_ht.squeeze())
            r_tt= self.R_ht_bn(r_tt.squeeze())

            dim_squeeze = 1
            r_ht = r_ht.unsqueeze(dim_squeeze)
            r_tt = r_tt.unsqueeze(dim_squeeze)
            if s is None:
                dim_squeeze = 0
            s_t = s_t.unsqueeze(dim_squeeze)
            dim_squeeze = 1
            if o is None:
                dim_squeeze = 0
            o_t = o_t.unsqueeze(dim_squeeze)

            '''
            print("o_t", o_t.shape)
            print("s_t", s_t.shape)
            print("r_ht", r_ht.shape)
            '''

        #'''
        ##Fast
        if s is None:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            head_type_compatibility = r_ht @ s_t.transpose(0,1)
        else:
            head_type_compatibility = (s_t*r_ht).sum(-1)
        if o is None:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            tail_type_compatibility = r_tt @ o_t.transpose(0,1)
        else:
            tail_type_compatibility = (o_t*r_tt).sum(-1)

        #'''

        '''
        ##using 1 neg samples for all facts in batch
        if s is None:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            head_type_compatibility = r_ht @ s_t.transpose(0,1)
        else:
            if s.shape[1] > 1:
                s_t = s_t[0].view(-1,self.embedding_dim)
                r_ht = r_ht.view(-1,self.embedding_dim)
                head_type_compatibility = r_ht @ s_t.transpose(0,1)
            else:
                head_type_compatibility = (s_t*r_ht).sum(-1)
        if o is None:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            tail_type_compatibility = r_tt @ o_t.transpose(0,1)
        else:
            if o.shape[1] > 1:
                o_t = o_t[0].view(-1,self.embedding_dim)
                r_tt = r_tt.view(-1,self.embedding_dim)
                tail_type_compatibility = r_tt @ o_t.transpose(0,1)
            else:
                tail_type_compatibility = (o_t*r_tt).sum(-1)
        '''

        if flag_debug:
            print("Before Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))

        #'''
        base_forward = nn.Sigmoid()(self.psi*base_forward)
        head_type_compatibility = nn.Sigmoid()(self.psi_h*head_type_compatibility)
        tail_type_compatibility = nn.Sigmoid()(self.psi_t*tail_type_compatibility)
        #'''

        #base_forward = nn.functional.relu(self.psi*base_forward)
        #head_type_compatibility = nn.functional.relu(self.psi_h*head_type_compatibility)
        #tail_type_compatibility = nn.functional.relu(self.psi_t*tail_type_compatibility)

        if flag_debug:
            print("After Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))

        return self.mult*base_forward*head_type_compatibility*tail_type_compatibility

    def regularizer(self, s, r, o, reg_val=0):
        if reg_val:
            self.reg = reg_val
        if self.reg == 2:
            return self.base_model.regularizer(s, r, o)
        elif self.reg ==3:
            '''
            s_t = self.E_t(s)
            r_ht = self.R_ht(r)
            r_tt = self.R_tt(r)
            o_t = self.E_t(o)

            factor = [torch.sqrt(s_t**2),torch.sqrt(o_t**2),torch.sqrt(r_ht**2+r_tt**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            #return self.type_reg*(reg/s.shape[0]) + self.base_reg*(self.base_model.regularizer(s, r, o, reg_val=self.reg))
            return (reg/s.shape[0]) + (self.base_model.regularizer(s, r, o, reg_val=self.reg))
            '''
            return self.base_model.regularizer(s, r, o, reg_val=self.reg)

    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
        return self.base_model.post_epoch()



class typed_model_lx(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, reg=2, flag_add_reverse=1):
        super(typed_model_lx, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count

        ##
        if flag_add_reverse:
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count
        self.flag_add_reverse=flag_add_reverse

        ##

        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.E_t = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_ht = nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt = nn.Embedding(self.relation_count, self.embedding_dim)
        nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        self.minimum_value = 0.0

        self.reg = reg

        print("Using reg", reg)
    def forward(self, s, r, o, flag_debug=0):
        base_forward = self.base_model(s, r, o)

        ##
        if self.flag_add_reverse:
            total_rel = torch.tensor(self.relation_count).cuda()
            inv_or_not = r >= total_rel; #inv_or_not = inv_or_not.type(torch.LongTensor)
            r = r - inv_or_not.type(torch.cuda.LongTensor) * total_rel
        ##

        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)

        '''#old
        head_type_compatibility = (s_t*r_ht).sum(-1)
        tail_type_compatibility = (o_t*r_tt).sum(-1)
        '''
        ##
        r_tt = r_tt.view(-1,self.embedding_dim)
        r_ht = r_ht.view(-1,self.embedding_dim)

        r_ht_new = torch.where(inv_or_not, r_tt, r_ht)
        r_tt_new = torch.where(inv_or_not, r_ht, r_tt)
        r_tt = r_tt_new; r_ht = r_ht_new; r_ht_new= None; r_tt_new=None

        r_ht = r_ht.unsqueeze(1)
        r_tt = r_tt.unsqueeze(1)
        ##
        #'''
        ##Fast
        if s is None:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            head_type_compatibility = r_ht @ s_t.transpose(0,1)
        else:
            head_type_compatibility = (s_t*r_ht).sum(-1)
        if o is None:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            tail_type_compatibility = r_tt @ o_t.transpose(0,1)
        else:
            tail_type_compatibility = (o_t*r_tt).sum(-1)

        #'''

        '''
        ##using 1 neg samples for all facts in batch
        if s is None:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            head_type_compatibility = r_ht @ s_t.transpose(0,1)
        else:
            if s.shape[1] > 1:
                s_t = s_t[0].view(-1,self.embedding_dim)
                r_ht = r_ht.view(-1,self.embedding_dim)
                head_type_compatibility = r_ht @ s_t.transpose(0,1)
            else:
                head_type_compatibility = (s_t*r_ht).sum(-1)
        if o is None:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            tail_type_compatibility = r_tt @ o_t.transpose(0,1)
        else:
            if o.shape[1] > 1:
                o_t = o_t[0].view(-1,self.embedding_dim)
                r_tt = r_tt.view(-1,self.embedding_dim)
                tail_type_compatibility = r_tt @ o_t.transpose(0,1)
            else:
                tail_type_compatibility = (o_t*r_tt).sum(-1)
        '''

        if flag_debug:
            print("Before Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))

        base_forward = nn.Sigmoid()(self.psi*base_forward)
        head_type_compatibility = nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = nn.Sigmoid()(self.psi*tail_type_compatibility)

        if flag_debug:
            print("After Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))

        return self.mult*base_forward*head_type_compatibility*tail_type_compatibility

    def regularizer(self, s, r, o, reg_val=0):
        if reg_val:
            self.reg = reg_val
        if self.reg == 2:
            return self.base_model.regularizer(s, r, o)
        elif self.reg ==3:
            '''
            s_t = self.E_t(s)
            r_ht = self.R_ht(r)
            r_tt = self.R_tt(r)
            o_t = self.E_t(o)

            factor = [torch.sqrt(s_t**2),torch.sqrt(o_t**2),torch.sqrt(r_ht**2+r_tt**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            #return self.type_reg*(reg/s.shape[0]) + self.base_reg*(self.base_model.regularizer(s, r, o, reg_val=self.reg))
            return (reg/s.shape[0]) + (self.base_model.regularizer(s, r, o, reg_val=self.reg))
            '''
            return self.base_model.regularizer(s, r, o, reg_val=self.reg)

    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
        return self.base_model.post_epoch()


class typed_model_v1(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, reg=2, flag_train_beta=1):
        super(typed_model_v1, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.E_t = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_ht = nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt = nn.Embedding(self.relation_count, self.embedding_dim)
        nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        self.minimum_value = 0.0

        self.reg = reg

        ##
        self.flag_train_beta = flag_train_beta
        if flag_train_beta:
            print("Training beta values")

        #better combination - convex
        self.beta = nn.Embedding(self.relation_count, 1)

        nn.init.constant_(self.beta.weight.data, 3.0)
        ##

    def forward(self, s, r, o, flag_debug=0):
        base_forward = self.base_model(s, r, o)
        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)

        '''#old
        head_type_compatibility = (s_t*r_ht).sum(-1)
        tail_type_compatibility = (o_t*r_tt).sum(-1)
        '''

        if s is None:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            head_type_compatibility = r_ht @ s_t.transpose(0,1)
        else:
            head_type_compatibility = (s_t*r_ht).sum(-1)
        if o is None:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            tail_type_compatibility = r_tt @ o_t.transpose(0,1)
        else:
            tail_type_compatibility = (o_t*r_tt).sum(-1)

        if flag_debug:
            print("Before Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))

        base_forward = nn.Sigmoid()(self.psi*base_forward)
        head_type_compatibility = nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = nn.Sigmoid()(self.psi*tail_type_compatibility)

        if flag_debug:
            print("After Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))


        if self.flag_train_beta==0:
            beta=1
        else:
            betas = self.beta(r).squeeze(2)
            beta = nn.Sigmoid()(betas)

        score_old = (base_forward*beta + 1.0 - beta)*head_type_compatibility*tail_type_compatibility

        return self.mult*score_old #base_forward*head_type_compatibility*tail_type_compatibility

    def regularizer(self, s, r, o, reg_val=0):
        if reg_val:
            self.reg = reg_val
        if self.reg == 2:
            return self.base_model.regularizer(s, r, o)
        elif self.reg ==3:
            '''
            s_t = self.E_t(s)
            r_ht = self.R_ht(r)
            r_tt = self.R_tt(r)
            o_t = self.E_t(o)

            factor = [torch.sqrt(s_t**2),torch.sqrt(o_t**2),torch.sqrt(r_ht**2+r_tt**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            #return self.type_reg*(reg/s.shape[0]) + self.base_reg*(self.base_model.regularizer(s, r, o, reg_val=self.reg))
            return (reg/s.shape[0]) + (self.base_model.regularizer(s, r, o, reg_val=self.reg))
            '''
            if self.flag_train_beta:
                beta = nn.Sigmoid()(self.beta(r))
                return self.base_model.regularizer(s, r, o, reg_val=self.reg) + (beta**2).sum()
            else:
                return self.base_model.regularizer(s, r, o, reg_val=self.reg)

    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
        return self.base_model.post_epoch()


class typed_model_reflexive(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, base_model_name, base_model_arguments, unit_reg=True, mult=20.0, psi=1.0, reg=2, flag_train_beta=0, flag_train_reflexive=0):
        super(typed_model_reflexive, self).__init__()

        base_model_class = globals()[base_model_name]
        base_model_arguments['entity_count'] = entity_count
        base_model_arguments['relation_count'] = relation_count
        self.base_model = base_model_class(**base_model_arguments)

        self.embedding_dim = embedding_dim
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.unit_reg = unit_reg
        self.mult = mult
        self.psi = psi
        self.E_t = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_ht = nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tt = nn.Embedding(self.relation_count, self.embedding_dim)
        nn.init.normal_(self.E_t.weight.data, 0, 0.05)
        nn.init.normal_(self.R_ht.weight.data, 0, 0.05)
        nn.init.normal_(self.R_tt.weight.data, 0, 0.05)
        self.minimum_value = 0.0

        self.reg = reg

        ##
        self.flag_train_beta = flag_train_beta
        if flag_train_beta:
            print("Training beta values")

        #better combination - convex
        self.beta = nn.Embedding(self.relation_count, 1)
        nn.init.constant_(self.beta.weight.data, 3.0)
        ##

        #reflexive
        self.flag_train_reflexive = flag_train_reflexive 
        if flag_train_reflexive:
            print("Training reflexive values")
        self.eps = nn.Embedding(self.relation_count, 1)
        nn.init.constant_(self.eps.weight.data, -3.0)
        ##


    def forward(self, s, r, o, flag_debug=0):
        base_forward = self.base_model(s, r, o)
        s_t = self.E_t(s) if s is not None else self.E_t.weight.unsqueeze(0)
        r_ht = self.R_ht(r)
        r_tt = self.R_tt(r)
        o_t = self.E_t(o) if o is not None else self.E_t.weight.unsqueeze(0)

        '''#old
        head_type_compatibility = (s_t*r_ht).sum(-1)
        tail_type_compatibility = (o_t*r_tt).sum(-1)
        '''

        if s is None:
            s_t = s_t.view(-1,self.embedding_dim)
            r_ht = r_ht.view(-1,self.embedding_dim)
            head_type_compatibility = r_ht @ s_t.transpose(0,1)
        else:
            head_type_compatibility = (s_t*r_ht).sum(-1)
        if o is None:
            o_t = o_t.view(-1,self.embedding_dim)
            r_tt = r_tt.view(-1,self.embedding_dim)
            tail_type_compatibility = r_tt @ o_t.transpose(0,1)
        else:
            tail_type_compatibility = (o_t*r_tt).sum(-1)

        if flag_debug:
            print("Before Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))

        base_forward = nn.Sigmoid()(self.psi*base_forward)
        head_type_compatibility = nn.Sigmoid()(self.psi*head_type_compatibility)
        tail_type_compatibility = nn.Sigmoid()(self.psi*tail_type_compatibility)

        if flag_debug:
            print("After Sigmoid")
            print("base_forward", torch.mean(base_forward), torch.std(base_forward))
            print("head_type_compatibility", torch.mean(head_type_compatibility), torch.std(head_type_compatibility))
            print("tail_type_compatibility", torch.mean(tail_type_compatibility), torch.std(tail_type_compatibility))

        ##convex combination
        if self.flag_train_beta==0:
            beta=1
        else:
            betas = self.beta(r).squeeze(2)
            beta = nn.Sigmoid()(betas)

        score_old = (base_forward*beta + 1.0 - beta)*head_type_compatibility*tail_type_compatibility
        ##

        ##reflexive
        if self.flag_train_reflexive == 0:
            score_new = score_old
        else:
            epsilon = self.eps(r).squeeze(2)
            eps = nn.Sigmoid()(epsilon)

            if s is None:
                base_o = score_old.gather(1, o)
                score_new = score_old.scatter_(1, o, base_o*eps)
                return score_new
            if o is None:
                base_s = score_old.gather(1, s)
                score_new = score_old.scatter_(1, s, base_s*eps)
                return score_new
 
            score_new = eps * (score_old * (s==o).type(torch.cuda.FloatTensor)) + (score_old * (s != o).type(torch.cuda.FloatTensor))
            if flag_debug:
                print("Debug: ", torch.mean(eps), torch.std(eps))
                a = torch.sum(eps * (s==o).type(torch.cuda.FloatTensor))
                b = torch.sum((s==o).type(torch.cuda.FloatTensor))
                print("Debug: ", a, b, a/b)
                a = torch.sum(eps * (s!=o).type(torch.cuda.FloatTensor))
                b = torch.sum((s!=o).type(torch.cuda.FloatTensor))
                print("Debug: ", a, b, a/b)
        ##

        return self.mult*score_new #base_forward*head_type_compatibility*tail_type_compatibility

    def regularizer(self, s, r, o, reg_val=0):
        if reg_val:
            self.reg = reg_val
        if self.reg == 2:
            return self.base_model.regularizer(s, r, o)
        elif self.reg ==3:
            '''
            s_t = self.E_t(s)
            r_ht = self.R_ht(r)
            r_tt = self.R_tt(r)
            o_t = self.E_t(o)

            factor = [torch.sqrt(s_t**2),torch.sqrt(o_t**2),torch.sqrt(r_ht**2+r_tt**2)]
            reg = 0
            for ele in factor:
                reg += torch.sum(torch.abs(ele) ** 3)
            #return self.type_reg*(reg/s.shape[0]) + self.base_reg*(self.base_model.regularizer(s, r, o, reg_val=self.reg))
            return (reg/s.shape[0]) + (self.base_model.regularizer(s, r, o, reg_val=self.reg))
            '''
            if self.flag_train_beta:
                beta = nn.Sigmoid()(self.beta(r))
                return self.base_model.regularizer(s, r, o, reg_val=self.reg) + (beta**2).sum()
            else:
                return self.base_model.regularizer(s, r, o, reg_val=self.reg) 

    def post_epoch(self):
        if(self.unit_reg):
            self.E_t.weight.data.div_(self.E_t.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_tt.weight.data.div_(self.R_tt.weight.data.norm(2, dim=-1, keepdim=True))
            self.R_ht.weight.data.div_(self.R_ht.weight.data.norm(2, dim=-1, keepdim=True))
        return self.base_model.post_epoch()


class DME(nn.Module):
    """
    DM+E model.
    deprecated. Use Adder model with DM and E as sub models for more control
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=False, clamp_v=None, display_norms=False):
        super(DME, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg

        self.E_DM = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_DM = nn.Embedding(self.relation_count, self.embedding_dim)

        nn.init.normal_(self.E_DM.weight.data, 0, 0.05)
        nn.init.normal_(self.R_DM.weight.data, 0, 0.05)

        self.E = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_head = nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tail = nn.Embedding(self.relation_count, self.embedding_dim)

        nn.init.normal_(self.E.weight.data, 0, 0.05)
        nn.init.normal_(self.R_head.weight.data, 0, 0.05)
        nn.init.normal_(self.R_tail.weight.data, 0, 0.05)

        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        self.diplay_norms=display_norms

    def forward(self, s, r, o):
        s_DM = self.E_DM(s) if s is not None else self.E_DM.weight.unsqueeze(0)
        r_DM = self.R_DM(r)
        o_DM = self.E_DM(o) if o is not None else self.E_DM.weight.unsqueeze(0)

        s = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r_head = self.R_head(r)
        r_tail = self.R_tail(r)
        o = self.E(o) if o is not None else self.E.weight.unsqueeze(0)

        if self.clamp_v:
            s.data.clamp_(-self.clamp_v, self.clamp_v)
            r_head.data.clamp_(-self.clamp_v, self.clamp_v)
            r_tail.data.clamp_(-self.clamp_v, self.clamp_v)
            o.data.clamp_(-self.clamp_v, self.clamp_v)

            s_DM.data.clamp_(-self.clamp_v, self.clamp_v)
            r_DM.data.clamp_(-self.clamp_v, self.clamp_v)

        out = (s*r_head+o*r_tail).sum(dim=-1) + (s_DM*r_DM*o_DM).sum(dim=-1)
        return out

    def regularizer(self, s, r, o):
        s_DM = self.E_DM(s)
        r_DM = self.R_DM(r)
        o_DM = self.E_DM(o)

        s = self.E(s)
        r_head = self.R_head(r)
        r_tail = self.R_tail(r)
        o = self.E(o)

        return (s*s+o*o+r_head*r_head+r_tail*r_tail+s_DM*s_DM+r_DM*r_DM+o_DM*o_DM).sum()#(s*s+o*o+r*r).sum()

    def post_epoch(self):
        e_norms = self.E.weight.data.norm(2, dim=-1, keepdim=True)
        r_head_norms = self.R_head.weight.data.norm(2, dim=-1)
        r_tail_norms = self.R_tail.weight.data.norm(2, dim=-1)

        max_e, min_e = torch.max(e_norms), torch.min(e_norms)
        max_r_tail, min_r_tail = torch.max(r_tail_norms), torch.min(r_tail_norms)
        max_r_head, min_r_head = torch.max(r_head_norms), torch.min(r_head_norms)

        e_DM_norms = self.E_DM.weight.data.norm(2, dim=-1, keepdim=True)
        r_DM_norms = self.R_DM.weight.data.norm(2, dim=-1)

        max_e_DM, min_e_DM = torch.max(e_DM_norms), torch.min(e_DM_norms)
        max_r_DM, min_r_DM = torch.max(r_DM_norms), torch.min(r_DM_norms)

        if self.unit_reg:
            self.E.weight.data.div_(e_norms)
            self.E_DM.weight.data.div_(e_DM_norms)
        if self.diplay_norms:
            return "(%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f)" % (max_e, min_e, max_r_head, min_r_head, max_r_tail, min_r_tail, max_e_DM, min_e_DM, max_r_DM, min_r_DM)
        else:
            return ""


class E(nn.Module):
    """
    E model \n
    scoring function (s, r, o) = s*r_h + o*r_o
    """
    def __init__(self, entity_count, relation_count, embedding_dim, unit_reg=True, clamp_v=None, display_norms=False):
        super(E, self).__init__()
        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.unit_reg = unit_reg

        self.E = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R_head = nn.Embedding(self.relation_count, self.embedding_dim)
        self.R_tail = nn.Embedding(self.relation_count, self.embedding_dim)
        nn.init.normal_(self.E.weight.data, 0, 0.05)
        nn.init.normal_(self.R_head.weight.data, 0, 0.05)
        nn.init.normal_(self.R_tail.weight.data, 0, 0.05)
        self.minimum_value = -self.embedding_dim*self.embedding_dim
        self.clamp_v = clamp_v
        self.display_norms = display_norms

    def forward(self, s, r, o):
        s = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r_head = self.R_head(r)
        r_tail = self.R_tail(r)
        o = self.E(o) if o is not None else self.E.weight.unsqueeze(0)
        if self.clamp_v:
            s.data.clamp_(-self.clamp_v, self.clamp_v)
            r_head.data.clamp_(-self.clamp_v, self.clamp_v)
            r_tail.data.clamp_(-self.clamp_v, self.clamp_v)
            o.data.clamp_(-self.clamp_v, self.clamp_v)
        return (s*r_head+o*r_tail).sum(dim=-1)

    def regularizer(self, s, r, o):
        s = self.E(s)
        r_head = self.R_head(r)
        r_tail = self.R_tail(r)
        o = self.E(o)
        return (s*s+o*o+r_head*r_head+r_tail*r_tail).sum()#(s*s+o*o+r*r).sum()

    def post_epoch(self):
        e_norms = self.E.weight.data.norm(2, dim=-1, keepdim=True)
        r_head_norms = self.R_head.weight.data.norm(2, dim=-1)
        r_tail_norms = self.R_tail.weight.data.norm(2, dim=-1)

        max_e, min_e = torch.max(e_norms), torch.min(e_norms)
        max_r_tail, min_r_tail = torch.max(r_tail_norms), torch.min(r_tail_norms)
        max_r_head, min_r_head = torch.max(r_head_norms), torch.min(r_head_norms)

        if self.unit_reg:
            self.E.weight.data.div_(e_norms)
        if self.display_norms:
            return "(%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f)" % (max_e, min_e, max_r_head, min_r_head, max_r_tail, min_r_tail)
        else:
            return ""



class transE(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim, clamp_v=None, reg=0, batch_norm=False, unit_reg=False, has_cuda=True, flag_add_reverse=0, flag_avg_scores=0):
        super(transE, self).__init__()

        self.entity_count = entity_count
        self.embedding_dim = embedding_dim
        self.relation_count = relation_count
        self.E = nn.Embedding(self.entity_count, self.embedding_dim)
        self.R = nn.Embedding(self.relation_count, self.embedding_dim)
        #nn.init.normal_(self.E.weight.data, 0, 0.05)
        #nn.init.normal_(self.R.weight.data, 0, 0.05)

        nn.init.xavier_normal(self.R.weight.data)
        nn.init.xavier_normal(self.E.weight.data)

        self.minimum_value = self.embedding_dim*self.embedding_dim #opposite for transE
        self.clamp_v = clamp_v

        self.unit_reg = unit_reg

        self.reg = reg
        print("Regularization value:", reg)

        if flag_add_reverse:
            print("transE: Added inverse relations!")
            self.relation_count = int(relation_count/2)
        else:
            self.relation_count = relation_count

        self.flag_add_reverse = flag_add_reverse

    def forward(self, s, r, o, flag_debug=0):
        s_e = self.E(s) if s is not None else self.E.weight.unsqueeze(0)
        r_e = self.R(r) if r is not None else self.R.weight.unsqueeze(0)
        o_e = self.E(o) if o is not None else self.E.weight.unsqueeze(0)

        if o is None or o.shape[1] > 1:
            result= torch.abs((s_e + r_e) - o_e).sum(dim=-1)
        elif s is None or s.shape[1] > 1:
            result= torch.abs((r_e - o_e) + s_e).sum(dim=-1)
        
        result= torch.abs(s_e + r_e - o_e).sum(dim=-1)
        return -result

    def regularizer(self, s, r, o, reg_val=0):
        s_e = self.E(s)
        r_e = self.R(r)
        o_e = self.E(o)
        if reg_val:
            self.reg = reg_val

        if self.reg == 2:
            return (s_e**2+o_e**2+r_e**2).sum()
        elif self.reg == 22:
            return (1.32e-7/2)*((s_e**2).sum() + (o_e**2).sum()) + (3.72e-18/2)*((r_e**2).sum())
        elif self.reg == 0:
            return None
        else:
            print("Unknown reg for TransE model")
            assert(False)


    def post_epoch(self):

        e_norms = self.E.weight.data.norm(2, dim=-1, keepdim=True)
        r_norms = self.R.weight.data.norm(2, dim=-1, keepdim=True)

        #max_e, min_e = torch.max(e_norms), torch.min(e_norms)
        #max_r, min_r = torch.max(r_norms), torch.min(r_norms)

        if self.unit_reg:
            self.E.weight.data.div_(e_norms)
            #self.R.weight.data.div_(r_norms)
        '''
        if self.diplay_norms:
            return "(%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f), (%.2f, %.2f)" % (max_e, min_e, max_r_head, min_r_head, max_r_tail, min_r_tail, max_e_DM, min_e_DM, max_r_DM, min_r_DM)
        else:
            return ""
        '''
        return ""
