import torch
import torch.nn as nn
import torch.nn.functional as F


class crossentropy_loss_bothNegSep(torch.nn.Module):
    def __init__(self):
        super(crossentropy_loss_bothNegSep, self).__init__()
        self.name = "crossentropy_loss_bothNegSep"
        self.loss = nn.CrossEntropyLoss(reduction = 'mean')
    def forward(self, positive, negative_1):
        scores = torch.cat([positive, negative_1], dim=-1)
        '''
        truth = torch.zeros(positive.shape[0],dtype=torch.long).cuda()#positive.shape[1]+negative_1.shape[1]+negative_2.shape[1]).cuda()
        #truth[:, 0] = 1
        '''
        ##
        truth = torch.zeros(positive.shape[0], dtype=torch.long).cuda()
        #truth[:, 0] = 1
        ##
        truth = torch.autograd.Variable(truth, requires_grad=False)
        losses = self.loss(scores, truth)
        return losses


class BCE_loss(torch.nn.Module):
    def __init__(self):
        super(BCE_loss, self).__init__()
        self.name = "crossentropy_loss_bothNegSep"
        self.loss = nn.BCELoss(reduction = 'mean')
    def forward(self, positive, negative_1):
        scores = torch.cat([positive, negative_1], dim=-1)
        truth = torch.zeros(scores.shape[0], scores.shape[1]).cuda()
        #truth[:, 0] = 1
        truth = torch.autograd.Variable(truth, requires_grad=False)
        #x = torch.log(1+torch.exp(-scores*truth))
        #total = x.sum()
        #return total/((positive.shape[1]+negative_1.shape[1])*positive.shape[0])
        #truth = truth.view(truth.shape[0])
        losses = self.loss(scores, truth)
        return losses


class softplus_loss(torch.nn.Module):
    def __init__(self):
        super(softplus_loss, self).__init__()
        self.name = "softplus_loss"
        self.loss = nn.BCELoss(reduction = 'mean')
    def forward(self, positive, negative):
        negative = negative.view(-1)
        
        positive =  positive.squeeze()
        negative = negative.squeeze()
 
        #scores = torch.cat([positive, negative_1], dim=-1)
        truth = torch.zeros(positive.shape[0]).cuda()
        false = torch.zeros(negative.shape[0]).cuda()
        truth[:] = 1
        false[:] = -1
        truth = torch.autograd.Variable(truth, requires_grad=False)
        false = torch.autograd.Variable(false, requires_grad=False)

        loss = torch.sum(F.softplus(-truth * positive))
        loss += torch.sum(F.softplus(-false * negative))

        #losses = self.loss(scores, truth)
        #pdb.set_trace()
        return loss#/(positive.shape[0]+negative.shape[0])#es

                #loss = torch.sum(F.softplus(-l * scores))+ (self.args.reg_lambda * self.model.l2_loss() / self.dataset.num_batch(self.args.batch_size))

class crossentropy_loss_AllNeg(torch.nn.Module):
    def __init__(self):
        super(crossentropy_loss_AllNeg, self).__init__()
        self.name = "crossentropy_loss_bothNegSep"
        self.loss = nn.CrossEntropyLoss(reduction = 'mean')
    def forward(self, truth, scores):
        # scores = torch.cat([positive, negative_1], dim=-1)
        # truth = torch.zeros(positive.shape[0],dtype=torch.long).cuda()
        #positive.shape[1]+negative_1.shape[1]+negative_2.shape[1]).cuda()
        #truth[:, 0] = 1
        #truth = torch.autograd.Variable(truth, requires_grad=False)

        
        # scores_truth = scores.gather(1,truth)
        # neg_samples_index = torch.from_numpy(numpy.random.randint(low=0,high=scores.shape[1]-1,size=(scores.shape[0],200))).cuda()
        # scores_false = scores.gather(1,neg_samples_index)
        # scores=""
        # scores_new = torch.cat([scores_truth, scores_false], dim=-1)
        

        # scores_new = torch.cat([scores.gather(1,truth),scores.gather(1,neg_samples_index)], dim=-1);
        #scores=""
        truth = torch.autograd.Variable(truth, requires_grad=False)  
        truth = truth.view(truth.shape[0])
        
        #pdb.set_trace()
        losses = self.loss(scores, truth)
        # losses = self.loss(scores_new, truth*0)
        # losses = self.loss(torch.cat([scores.gather(1,truth),scores.gather(1,neg_samples_index)], dim=-1), truth*0)
        #losses = self.loss(torch.cat([scores.gather(1,truth),scores.gather(1,torch.from_numpy(numpy.random.randint(low=0,high=scores.shape[1]-1,size=(scores.shape[0],10))).cuda())], dim=-1), truth*0)
        return losses

class Alignment_Loss(torch.nn.Module):
    def __init__(self):
        super(crossentropy_loss_AllNeg_subsample, self).__init__()
        self.name = "crossentropy_loss_AllNeg_subsample"
        self.loss = nn.CrossEntropyLoss(reduction = 'mean')

    def forward(self, truth, scores, negative_count):
        truth = torch.autograd.Variable(truth, requires_grad=False)  
        #truth = truth.view(truth.shape[0])
        #pdb.set_trace()


        positive = scores.gather(1,truth.view(-1,1))
        negative = scores[:,torch.randperm(scores.size(1))[:negative_count]]

        scores = torch.cat([positive, negative], dim=-1)
        truth = torch.zeros(positive.shape[0], dtype=torch.long).cuda()

        #weight = torch.ones_like(truth)

        losses = self.loss(scores, truth)
        return losses
class crossentropy_loss_AllNeg_subsample(torch.nn.Module):
    def __init__(self):
        super(crossentropy_loss_AllNeg_subsample, self).__init__()
        self.name = "crossentropy_loss_AllNeg_subsample"
        self.loss = nn.CrossEntropyLoss(reduction = 'mean')

    def forward(self, truth, scores, negative_count):
        truth = torch.autograd.Variable(truth, requires_grad=False)  
        #truth = truth.view(truth.shape[0])
        #pdb.set_trace()


        positive = scores.gather(1,truth.view(-1,1))
        negative = scores[:,torch.randperm(scores.size(1))[:negative_count]]

        scores = torch.cat([positive, negative], dim=-1)
        truth = torch.zeros(positive.shape[0], dtype=torch.long).cuda()

        #weight = torch.ones_like(truth)

        losses = self.loss(scores, truth)
        return losses



class softmax_loss_wtpos(torch.nn.Module):
    def __init__(self):
        super(softmax_loss_wtpos, self).__init__()
    def forward(self, positive, negative_1):#, negative_2):
        negative = torch.cat([positive, negative_1],dim=-1)#, negative_2], dim=-1)
        max_den = negative.max(dim=1, keepdim=True)[0].detach()
        den = (negative-max_den).exp().sum(dim=-1, keepdim=True)
        losses = (positive-max_den) - den.log()

        return - (losses.mean())

class softmax_loss_wopos(torch.nn.Module):
    def __init__(self):
        super(softmax_loss_wopos, self).__init__()
        self.name = "softmax_loss_wopos"
    def forward(self, positive, negative):#, negative_2):
        #negative = torch.cat([positive, negative_1],dim=-1)#, negative_2], dim=-1)
        max_den = negative.max(dim=1, keepdim=True)[0].detach()
        den = (negative-max_den).exp().sum(dim=-1, keepdim=True)
        losses = (positive-max_den) - den.log()

        return - (losses.mean())

class softmax_loss_AllNeg(torch.nn.Module):
    def __init__(self):
        super(softmax_loss_AllNeg, self).__init__()
    def forward(self, positive, scores):
        #pdb.set_trace()
        #negative = torch.cat([positive, negative_1],dim=-1)#, negative_2], dim=-1)
        max_den = scores.max(dim=1, keepdim=True)[0].detach()
        den = (scores-max_den).exp().sum(dim=-1, keepdim=True)
        #print("positive",positive.shape,"scores", scores.shape, "max_den", max_den.shape)
        truth = positive.view(positive.shape[0])
        losses = (scores[:,truth]-max_den) - den.log()

        return - (losses.mean())


class test(torch.nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction = 'mean')

    def forward(self, positive, negative_1, negative_2):
        #'''
        max_den_e1 = negative_1.max(dim=1, keepdim=True)[0].detach()
        ##max_den_e2 = negative_2.max(dim=1, keepdim=True)[0].detach()
        #print("max_den_e1",max_den_e1.shape)
        #print("(negative_1-max_den_e1)",(negative_1-max_den_e1).shape)
        den_e1 = (negative_1-max_den_e1).exp().sum(dim=-1, keepdim=True)
        ##den_e2 = (negative_2-max_den_e2).exp().sum(dim=-1, keepdim=True)
        #print("den_e1",den_e1.shape)
        ##losses = ((2*positive-max_den_e1-max_den_e2) - den_e1.log() - den_e2.log())
        losses = ((positive-max_den_e1) - den_e1.log())
        #print("positive-max_den_e1", (positive-max_den_e1).shape)
        #print("(positive-max_den_e1)-den_e1.log()",((positive-max_den_e1)-den_e1.log()).shape)
        #'''
        den_e1_noOverflow = (negative_1).exp().sum(dim=-1, keepdim=True)
        losses_noOverflow = ((positive) - den_e1_noOverflow.log())

        ##
        #part 1c
        scores_denPos = torch.cat([positive, negative_1], dim=-1)
        max_den_e1_denPos = scores_denPos.max(dim=1, keepdim=True)[0].detach()
        den_e1_denPos = (scores_denPos-max_den_e1_denPos).exp().sum(dim=-1, keepdim=True)
        losses_denPos = ((positive-max_den_e1_denPos) - den_e1_denPos.log())
        #

        ##part2
        ##scores = torch.cat([positive, negative_1, negative_2], dim=-1)
        scores = torch.cat([positive, negative_1], dim=-1)
        print("scores for pre-built functions", scores.shape)
        truth = torch.zeros(positive.shape[0],dtype=torch.long).cuda()#positive.shape[1]+negative_1.shape[1]+negative_2.shape[1]).cuda()
        #truth[:, 0] = 1
        truth = torch.autograd.Variable(truth, requires_grad=False)
        losses_ce = self.loss(scores, truth)

        #part3
        loss1 = nn.LogSoftmax(dim=1)
        loss2 = nn.NLLLoss()
        scores3 = loss1(scores)
        losses_ce_2 = loss2(scores, truth)

        ##
        print("!!!")
        print("nn.LogSoftmax", scores3[:,0], scores3[:,0].shape)
        print("manual log softMax", losses, losses.shape)
        print("manual log softMax noOverflow", losses_noOverflow, losses_noOverflow.shape)
        print("manual log softMax with denPos", losses_denPos, losses_denPos.shape)
        print("!!!")
        ##
        print("CE loss:", losses_ce)
        print("SM loss:", -losses.mean())
        print("SM noOverflow loss:", -losses_noOverflow.mean())
        print("SM denPos loss:", -losses_denPos.mean())
        print("CE 2 loss:", losses_ce_2)

        return -losses.mean()

class logistic_loss(torch.nn.Module):
    def __init__(self):
        super(logistic_loss, self).__init__()
        self.name = "logistic_loss"
    def forward(self, positive, negative_1):
        scores = torch.cat([positive, negative_1], dim=-1)
        truth = torch.ones(1, positive.shape[1]+negative_1.shape[1]).cuda()
        truth[0, 0] = -1
        truth = -truth
        truth = torch.autograd.Variable(truth, requires_grad=False)
        x = torch.log(1+torch.exp(-scores*truth))
        total = x.sum()
        return total/((positive.shape[1]+negative_1.shape[1])*positive.shape[0])


class hinge_loss(torch.nn.Module):
    def __init__(self):
        super(hinge_loss, self).__init__()
        self.name = "hinge_loss"
    def forward(self, positive, negative_1):
        scores = torch.cat([positive, negative_1], dim=-1)
        truth = torch.ones(1, positive.shape[1]+negative_1.shape[1]).cuda()
        #truth = torch.ones(positive.shape[0], positive.shape[1]+negative_1.shape[1]).cuda()

        #pdb.set_trace()
        truth[0, 0] = -1
        #truth[:, 0] = -1
        truth = -truth
        truth = torch.autograd.Variable(truth, requires_grad=False)

        return nn.HingeEmbeddingLoss(margin=4)(scores, truth)

'''
class crossentropy_loss(torch.nn.Module):
    def __init__(self):
        super(crossentropy_loss, self).__init__()
        self.name = "crossentropy_loss"
        self.loss = nn.CrossEntropyLoss(reduction = 'mean')
    def forward(self, positive, negative_1, negative_2):
        scores = torch.cat([positive, negative_1, negative_2], dim=-1)
        truth = torch.zeros(positive.shape[0],dtype=torch.long).cuda()#positive.shape[1]+negative_1.shape[1]+negative_2.shape[1]).cuda()
        #truth[:, 0] = 1
        truth = torch.autograd.Variable(truth, requires_grad=False)
        losses = self.loss(scores, truth)
        return losses

class crossentropy_loss_bothNegSep(torch.nn.Module):
    def __init__(self):
        super(crossentropy_loss_bothNegSep, self).__init__()
        self.name = "crossentropy_loss"
        self.loss = nn.CrossEntropyLoss(reduction = 'mean')
    def forward(self, positive, negative_1, negative_2):
        scores = torch.cat([positive, negative_1], dim=-1)
        truth = torch.zeros(positive.shape[0],dtype=torch.long).cuda()#positive.shape[1]+negative_1.shape[1]+negative_2.shape[1]).cuda()
        #truth[:, 0] = 1
        truth = torch.autograd.Variable(truth, requires_grad=False)
        losses_1 = self.loss(scores, truth)

        scores_2 = torch.cat([positive, negative_2], dim=-1)
        truth_2 = torch.zeros(positive.shape[0],dtype=torch.long).cuda()#positive.shape[1]+negative_1.shape[1]+negative_2.shape[1]).cuda()
        #truth[:, 0] = 1
        truth_2 = torch.autograd.Variable(truth_2, requires_grad=False)
        losses_2 = self.loss(scores_2, truth_2)

        losses = losses_1 + losses_2

        return losses



class softmax_loss(torch.nn.Module):
    def __init__(self):
        super(softmax_loss, self).__init__()
    def forward(self, positive, negative_1, negative_2):
        max_den_e1 = negative_1.max(dim=1, keepdim=True)[0].detach()
        max_den_e2 = negative_2.max(dim=1, keepdim=True)[0].detach()
        #print("max_den_e1",max_den_e1.shape)
        #print("(negative_1-max_den_e1)",(negative_1-max_den_e1).shape)
        den_e1 = (negative_1-max_den_e1).exp().sum(dim=-1, keepdim=True)
        den_e2 = (negative_2-max_den_e2).exp().sum(dim=-1, keepdim=True)
        #print("den_e1",den_e1.shape)
        losses = ((2*positive-max_den_e1-max_den_e2) - den_e1.log() - den_e2.log())
        #print("positive-max_den_e1", (positive-max_den_e1).shape)
        #print("(positive-max_den_e1)-den_e1.log()",((positive-max_den_e1)-den_e1.log()).shape)
        return -losses.mean()

class softmax_loss_wtpos(torch.nn.Module):
    def __init__(self):
        super(softmax_loss_wtpos, self).__init__()
    def forward(self, positive, negative_1, negative_2):
        negative = torch.cat([positive, negative_1],dim=-1)#, negative_2], dim=-1)
        max_den = negative.max(dim=1, keepdim=True)[0].detach()
        den = (negative-max_den).exp().sum(dim=-1, keepdim=True)
        losses = (positive-max_den) - den.log()

        ##
        negative_v2 = torch.cat([positive, negative_2],dim=-1)#, negative_2], dim=-1)
        max_den_v2  = negative_v2.max(dim=1, keepdim=True)[0].detach()
        den_v2  = (negative_v2 -max_den_v2 ).exp().sum(dim=-1, keepdim=True)
        losses_v2  = (positive-max_den_v2) - den_v2.log()


        return - (losses.mean() + losses_v2.mean())

class logistic_loss(torch.nn.Module):
    def __init__(self):
        super(logistic_loss, self).__init__()
    def forward(self, positive, negative_1, negative_2):
        scores = torch.cat([positive, negative_1, negative_2], dim=-1)
        truth = torch.ones(1, positive.shape[1]+negative_1.shape[1]+negative_2.shape[1]).cuda()
        truth[0, 0] = -1
        truth = -truth
        truth = torch.autograd.Variable(truth, requires_grad=False)
        x = torch.log(1+torch.exp(-scores*truth))
        total = x.sum()
        return total/((positive.shape[1]+negative_1.shape[1]+negative_2.shape[1])*positive.shape[0])
'''
